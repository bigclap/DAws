use std::collections::HashSet;

use crate::diffusion::cosine_similarity;
use crate::signal::Network;

#[derive(Clone, Copy, Debug)]
pub struct RewardConfig {
    pub alpha: f32,
    pub beta: f32,
    pub gamma: f32,
    pub activation_threshold: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct RewardBreakdown {
    pub reward: f32,
    pub logic: f32,
    pub sparsity: f32,
    pub consistency: f32,
}

pub struct RewardCalculator {
    config: RewardConfig,
    previous_state: Option<Vec<f32>>,
}

impl RewardCalculator {
    pub fn new(config: RewardConfig) -> Self {
        Self {
            config,
            previous_state: None,
        }
    }

    pub fn evaluate(
        &mut self,
        network: &Network,
        expected_active: &[usize],
        actual_spikes: &[usize],
    ) -> RewardBreakdown {
        let logic = logic_score(expected_active, actual_spikes);
        let sparsity = network.active_ratio(self.config.activation_threshold);
        let state = network.state_vector();
        let consistency = if let Some(previous) = &self.previous_state {
            cosine_similarity(previous, &state)
        } else {
            1.0
        };
        self.previous_state = Some(state);
        let reward = self.config.alpha * logic - self.config.beta * sparsity
            + self.config.gamma * consistency;
        RewardBreakdown {
            reward,
            logic,
            sparsity,
            consistency,
        }
    }
}

fn logic_score(expected: &[usize], actual: &[usize]) -> f32 {
    let expected_set: HashSet<_> = expected.iter().copied().collect();
    let actual_set: HashSet<_> = actual.iter().copied().collect();
    if expected_set.is_empty() {
        return if actual_set.is_empty() { 1.0 } else { -1.0 };
    }
    let true_positive = expected_set.intersection(&actual_set).count() as f32;
    let false_positive = actual_set.difference(&expected_set).count() as f32;
    let false_negative = expected_set.difference(&actual_set).count() as f32;
    let denom = expected_set.len() as f32;
    ((true_positive - false_positive - false_negative) / denom).clamp(-1.0, 1.0)
}
