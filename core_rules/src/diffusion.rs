use std::cmp::Ordering;

use core_graph::Network;
use retriever::Retriever;

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
pub struct FactRecruitment {
    retriever: Retriever,
    facts: Vec<Vec<f32>>,
    strength: f32,
    temperature: f32,
}

impl FactRecruitment {
    pub fn new(
        retriever: Retriever,
        facts: Vec<Vec<f32>>,
        strength: f32,
        temperature: f32,
    ) -> Self {
        Self {
            retriever,
            facts,
            strength: strength.max(0.0),
            temperature: temperature.max(1e-3),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.facts.is_empty()
    }

    pub fn recruit(&self, query: &[f32]) -> Option<Vec<f32>> {
        if self.is_empty() || query.is_empty() {
            return None;
        }
        let mut blended = vec![0.0; query.len()];
        let mut weight_sum = 0.0f32;
        for (idx, similarity) in self.approximate_candidates(query) {
            let fact = self.facts.get(idx)?;
            if fact.len() != query.len() {
                continue;
            }
            let weight = ((similarity + 1.0) * 0.5)
                .clamp(0.0, 1.0)
                .powf(1.0 / self.temperature);
            if weight <= 0.0 {
                continue;
            }
            weight_sum += weight;
            for (slot, &value) in blended.iter_mut().zip(fact.iter()) {
                *slot += weight * value;
            }
        }
        if weight_sum <= 0.0 {
            return None;
        }
        for value in &mut blended {
            *value = (*value / weight_sum) * self.strength;
        }
        Some(blended)
    }

    fn approximate_candidates(&self, query: &[f32]) -> Vec<(usize, f32)> {
        let total = self.facts.len();
        if total == 0 {
            return Vec::new();
        }
        let top_k = self.retriever.top_k().max(1);
        let step = (total / (top_k * 2)).max(1);
        let mut scored: Vec<_> = (0..total)
            .step_by(step)
            .filter_map(|idx| {
                let fact = self.facts.get(idx)?;
                (fact.len() == query.len()).then(|| (idx, cosine_similarity(query, fact)))
            })
            .collect();
        if scored.is_empty() {
            return scored;
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        scored.truncate(top_k);
        if step == 1 {
            return scored;
        }
        let mut refined = scored.clone();
        let window = step.min(4);
        for &(center, _) in &scored {
            let start = center.saturating_sub(window);
            let end = (center + window).min(total - 1);
            for idx in start..=end {
                if refined.iter().any(|(seen, _)| *seen == idx) {
                    continue;
                }
                if let Some(fact) = self.facts.get(idx) {
                    if fact.len() == query.len() {
                        refined.push((idx, cosine_similarity(query, fact)));
                    }
                }
            }
        }
        refined.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        refined.truncate(top_k);
        refined
    }
}

#[derive(Clone, Debug)]
pub struct DiffusionConfig {
    pub alpha_schedule: AnnealingSchedule,
    pub sigma_schedule: AnnealingSchedule,
    pub tolerance: f32,
    pub max_iters: usize,
    pub entropy_policy: EntropyPolicy,
    pub fact_recruitment: Option<FactRecruitment>,
}

#[derive(Debug, Clone)]
pub struct DiffusionOutcome {
    pub state: Vec<f32>,
}

pub struct DiffusionLoop {
    config: DiffusionConfig,
    last_similarity: f32,
    last_iterations: usize,
}

impl DiffusionLoop {
    pub fn new(config: DiffusionConfig) -> Self {
        Self {
            config,
            last_similarity: 0.0,
            last_iterations: 0,
        }
    }

    pub fn run(&mut self, network: &mut Network) -> DiffusionOutcome {
        let mut current = network.state_vector();
        if current.is_empty() {
            self.last_similarity = 1.0;
            self.last_iterations = 0;
            return DiffusionOutcome { state: current };
        }
        let mut last_similarity = 1.0f32;
        for iter in 0..self.config.max_iters {
            let consensus = network.consensus_state();
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
            network.set_state(&next);
            current = next;
            if 1.0 - last_similarity <= self.config.tolerance {
                self.last_iterations = iter + 1;
                self.last_similarity = last_similarity;
                return DiffusionOutcome {
                    state: network.state_vector(),
                };
            }
        }
        self.last_iterations = self.config.max_iters;
        self.last_similarity = last_similarity;
        DiffusionOutcome {
            state: network.state_vector(),
        }
    }

    pub fn last_similarity(&self) -> f32 {
        self.last_similarity
    }

    pub fn last_iterations(&self) -> usize {
        self.last_iterations
    }
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
