use anyhow::{Result, ensure};

#[derive(Clone, Debug, PartialEq)]
pub struct OnlinePlasticityConfig {
    pub update_interval: usize,
    pub structure_interval: usize,
    pub prune_threshold: f32,
    pub grow_threshold: f32,
    pub max_trace_log: usize,
}

impl Default for OnlinePlasticityConfig {
    fn default() -> Self {
        Self {
            update_interval: 8,
            structure_interval: 128,
            prune_threshold: 1e-3,
            grow_threshold: 0.5,
            max_trace_log: 64,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TraceLogEntry {
    pub step: usize,
    pub trace: Vec<f32>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PlasticityStepOutcome {
    pub applied_update: bool,
    pub pruned: usize,
    pub grown: usize,
}

pub struct OnlinePlasticity {
    config: OnlinePlasticityConfig,
    weights: Vec<f32>,
    step: usize,
    eligibility_buffer: Vec<Vec<f32>>,
    trace_log: Vec<TraceLogEntry>,
}

impl OnlinePlasticity {
    pub fn new(initial_weights: Vec<f32>, config: OnlinePlasticityConfig) -> Self {
        Self {
            config,
            weights: initial_weights,
            step: 0,
            eligibility_buffer: Vec::new(),
            trace_log: Vec::new(),
        }
    }

    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    pub fn trace_log(&self) -> &[TraceLogEntry] {
        &self.trace_log
    }

    pub fn step(&mut self, trace: Vec<f32>, reward: f32) -> Result<PlasticityStepOutcome> {
        ensure!(
            trace.len() == self.weights.len(),
            "trace length must match weights"
        );

        self.step = self.step.saturating_add(1);
        self.eligibility_buffer.push(trace.clone());
        self.push_trace_log(trace);

        let mut outcome = PlasticityStepOutcome::default();

        if self.config.update_interval > 0 && self.step % self.config.update_interval == 0 {
            outcome.applied_update = self.apply_reward_modulated_update(reward);
        }

        if self.config.structure_interval > 0 && self.step % self.config.structure_interval == 0 {
            let (pruned, grown) = self.apply_structural_plasticity(reward);
            outcome.pruned = pruned;
            outcome.grown = grown;
        }

        Ok(outcome)
    }

    fn apply_reward_modulated_update(&mut self, reward: f32) -> bool {
        if self.eligibility_buffer.is_empty() {
            return false;
        }

        let mut accumulated = vec![0.0f32; self.weights.len()];
        for trace in &self.eligibility_buffer {
            for (acc, value) in accumulated.iter_mut().zip(trace) {
                *acc += *value;
            }
        }
        let scale = reward / self.eligibility_buffer.len() as f32;
        for (weight, delta) in self.weights.iter_mut().zip(accumulated) {
            *weight += delta * scale;
        }

        self.eligibility_buffer.clear();
        true
    }

    fn apply_structural_plasticity(&mut self, reward: f32) -> (usize, usize) {
        let mut pruned = 0usize;
        for weight in self.weights.iter_mut() {
            if weight.abs() < self.config.prune_threshold {
                *weight = 0.0;
                pruned += 1;
            }
        }

        let mut grown = 0usize;
        if reward.abs() >= self.config.grow_threshold {
            let direction = if reward >= 0.0 { 1.0 } else { -1.0 };
            self.weights.push(direction * self.config.grow_threshold);
            grown = 1;
        }

        (pruned, grown)
    }

    fn push_trace_log(&mut self, trace: Vec<f32>) {
        if self.config.max_trace_log == 0 {
            return;
        }

        self.trace_log.push(TraceLogEntry {
            step: self.step,
            trace,
        });

        if self.trace_log.len() > self.config.max_trace_log {
            let overflow = self.trace_log.len() - self.config.max_trace_log;
            self.trace_log.drain(0..overflow);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reward_updates_are_scaled_by_reward() {
        let config = OnlinePlasticityConfig {
            update_interval: 1,
            structure_interval: 100,
            prune_threshold: 0.0,
            grow_threshold: 10.0,
            max_trace_log: 4,
        };
        let mut plasticity = OnlinePlasticity::new(vec![0.5, 0.5], config);

        plasticity.step(vec![1.0, 1.0], 0.5).unwrap();
        assert!((plasticity.weights()[0] - 1.0).abs() < 1e-6);
        assert!((plasticity.weights()[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn structural_plasticity_applies_thresholds() {
        let config = OnlinePlasticityConfig {
            update_interval: 1,
            structure_interval: 2,
            prune_threshold: 0.2,
            grow_threshold: 0.3,
            max_trace_log: 4,
        };
        let mut plasticity = OnlinePlasticity::new(vec![0.1, 0.5], config);

        plasticity.step(vec![0.0, 0.0], 0.0).unwrap();
        let outcome = plasticity.step(vec![0.0, 0.0], 0.4).unwrap();

        assert_eq!(outcome.pruned, 1);
        assert_eq!(outcome.grown, 1);
        assert_eq!(plasticity.weights().len(), 3);
    }

    #[test]
    fn trace_logs_respect_capacity() {
        let config = OnlinePlasticityConfig {
            update_interval: 10,
            structure_interval: 10,
            prune_threshold: 0.0,
            grow_threshold: 10.0,
            max_trace_log: 1,
        };
        let mut plasticity = OnlinePlasticity::new(vec![0.0], config);

        plasticity.step(vec![0.1], 0.0).unwrap();
        plasticity.step(vec![0.2], 0.0).unwrap();

        assert_eq!(plasticity.trace_log().len(), 1);
        assert_eq!(plasticity.trace_log()[0].step, 2);
    }
}
