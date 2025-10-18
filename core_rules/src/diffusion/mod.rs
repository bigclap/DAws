use core_graph::Network;
mod fact_recruitment;
pub use fact_recruitment::FactRecruitment;

#[derive(Clone, Debug)]
pub struct AnnealingSchedule {
    values: Vec<f32>,
}

impl AnnealingSchedule {
    pub fn new(values: Vec<f32>) -> Self {
        Self { values }
    }
    pub fn constant(value: f32) -> Self {
        Self {
            values: vec![value],
        }
    }
    pub fn value_at(&self, step: usize) -> f32 {
        match self.values.as_slice() {
            [] => 0.0,
            values => values[step.min(values.len().saturating_sub(1))],
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct EntropyPolicy {
    pub low_threshold: f32,
    pub high_threshold: f32,
    pub boost: f32,
    pub dampening: f32,
}

impl Default for EntropyPolicy {
    fn default() -> Self {
        Self {
            low_threshold: 0.35,
            high_threshold: 0.85,
            boost: 0.5,
            dampening: 0.35,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DiffusionConfig {
    pub alpha_schedule: AnnealingSchedule,
    pub sigma_schedule: AnnealingSchedule,
    pub tolerance: f32,
    pub jt_tolerance: f32,
    pub stability_tolerance: f32,
    pub stability_window: usize,
    pub max_energy_increase: usize,
    pub max_iters: usize,
    pub entropy_policy: EntropyPolicy,
    pub fact_recruitment: Option<FactRecruitment>,
}

#[derive(Debug, Clone, Copy)]
pub struct DiffusionDiagnostics {
    pub iterations: usize,
    pub similarity: f32,
    pub jt: f32,
    pub stability_streak: usize,
    pub energy: f32,
    pub energy_monotonic: bool,
}

impl Default for DiffusionDiagnostics {
    fn default() -> Self {
        Self {
            iterations: 0,
            similarity: 1.0,
            jt: 0.0,
            stability_streak: 0,
            energy: 0.0,
            energy_monotonic: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DiffusionOutcome {
    pub state: Vec<f32>,
    pub diagnostics: DiffusionDiagnostics,
}

pub struct DiffusionLoop {
    config: DiffusionConfig,
    diagnostics: DiffusionDiagnostics,
}

impl DiffusionLoop {
    pub fn new(config: DiffusionConfig) -> Self {
        Self {
            config,
            diagnostics: DiffusionDiagnostics::default(),
        }
    }

    pub fn run(&mut self, network: &mut Network) -> DiffusionOutcome {
        let mut current = network.state_vector();
        if current.is_empty() {
            self.diagnostics = DiffusionDiagnostics {
                iterations: 0,
                similarity: 1.0,
                jt: 0.0,
                stability_streak: 0,
                energy: network.energy(),
                energy_monotonic: true,
            };
            return DiffusionOutcome {
                state: current,
                diagnostics: self.diagnostics,
            };
        }
        let mut last_similarity = 1.0f32;
        let mut stability_streak = 0usize;
        let mut last_energy = network.energy();
        let mut energy_increase_streak = 0usize;
        let mut energy_monotonic = true;
        let mut last_jt = 0.0f32;
        for iter in 0..self.config.max_iters {
            let consensus = network.consensus_state();
            last_jt = jt_metric(&current, &consensus);
            let alpha = apply_entropy_policy(
                self.config.alpha_schedule.value_at(iter),
                normalised_entropy(&current),
                &self.config.entropy_policy,
            );
            let sigma = self.config.sigma_schedule.value_at(iter);
            let injection_vec = self
                .config
                .fact_recruitment
                .as_ref()
                .and_then(|recruiter| recruiter.recruit(&current));
            let injection = injection_vec.as_ref().filter(|vec| !vec.is_empty());
            let mut next = Vec::with_capacity(consensus.len());
            for (idx, (value, consensus_value)) in current.iter().zip(consensus.iter()).enumerate()
            {
                let recruit_term = injection
                    .and_then(|vec| vec.get(idx))
                    .copied()
                    .unwrap_or(0.0);
                let noise = deterministic_noise(idx, iter, sigma);
                next.push(
                    (value + alpha * (consensus_value - value) + recruit_term + noise)
                        .clamp(0.0, 1.0),
                );
            }
            last_similarity = cosine_similarity(&current, &next);
            let delta = stability_delta(&current, &next);
            if delta <= self.config.stability_tolerance {
                stability_streak = stability_streak.saturating_add(1);
            } else {
                stability_streak = 0;
            }
            network.set_state(&next);
            current = next;
            let energy = network.energy();
            if energy > last_energy + 1e-6 {
                energy_increase_streak = energy_increase_streak.saturating_add(1);
                energy_monotonic = false;
            } else {
                energy_increase_streak = 0;
            }
            last_energy = energy;

            let reached_similarity = 1.0 - last_similarity <= self.config.tolerance;
            let reached_jt = self.config.jt_tolerance > 0.0 && last_jt <= self.config.jt_tolerance;
            let reached_stability = self.config.stability_window > 0
                && stability_streak >= self.config.stability_window;
            let energy_violation = self.config.max_energy_increase > 0
                && energy_increase_streak >= self.config.max_energy_increase;

            if reached_similarity || reached_jt || reached_stability || energy_violation {
                let diagnostics = DiffusionDiagnostics {
                    iterations: iter + 1,
                    similarity: last_similarity,
                    jt: last_jt,
                    stability_streak,
                    energy,
                    energy_monotonic,
                };
                self.diagnostics = diagnostics;
                return DiffusionOutcome {
                    state: network.state_vector(),
                    diagnostics,
                };
            }
        }
        let diagnostics = DiffusionDiagnostics {
            iterations: self.config.max_iters,
            similarity: last_similarity,
            jt: last_jt,
            stability_streak,
            energy: last_energy,
            energy_monotonic,
        };
        self.diagnostics = diagnostics;
        DiffusionOutcome {
            state: network.state_vector(),
            diagnostics,
        }
    }

    pub fn last_similarity(&self) -> f32 {
        self.diagnostics.similarity
    }

    pub fn last_iterations(&self) -> usize {
        self.diagnostics.iterations
    }

    pub fn diagnostics(&self) -> DiffusionDiagnostics {
        self.diagnostics
    }
}

fn jt_metric(current: &[f32], consensus: &[f32]) -> f32 {
    if current.is_empty() || current.len() != consensus.len() {
        return 0.0;
    }
    let sum: f32 = current
        .iter()
        .zip(consensus.iter())
        .map(|(a, b)| {
            let diff = a - b;
            diff * diff
        })
        .sum();
    sum / current.len() as f32
}

fn stability_delta(current: &[f32], next: &[f32]) -> f32 {
    current
        .iter()
        .zip(next.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}

fn deterministic_noise(index: usize, iteration: usize, amplitude: f32) -> f32 {
    if amplitude == 0.0 {
        return 0.0;
    }
    let phase = (index as f32 * 0.618_033_9 + iteration as f32 * 0.414_213_56).sin();
    phase * amplitude
}

fn normalised_entropy(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let epsilon = 1e-6;
    let magnitude: f32 = values.iter().map(|v| v.abs()).sum();
    if magnitude <= epsilon {
        return 0.0;
    }
    let denom = magnitude + epsilon * values.len() as f32;
    let entropy = values
        .iter()
        .map(|v| {
            let p = (v.abs() + epsilon) / denom;
            -p * p.ln()
        })
        .sum::<f32>();
    let max_entropy = (values.len() as f32).ln();
    if max_entropy <= 0.0 {
        0.0
    } else {
        (entropy / max_entropy).clamp(0.0, 1.0)
    }
}

fn apply_entropy_policy(alpha: f32, entropy: f32, policy: &EntropyPolicy) -> f32 {
    let mut scaled = alpha.max(0.0);
    if entropy < policy.low_threshold {
        scaled *= 1.0 + policy.boost.max(0.0);
    }
    if entropy > policy.high_threshold {
        scaled *= 1.0 - policy.dampening.clamp(0.0, 0.99);
    }
    scaled
}
#[cfg(test)]
mod tests;
