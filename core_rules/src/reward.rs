//! Reward shaping utilities that integrate sparsity and consistency signals.

use std::collections::HashSet;

use core_graph::Network;

use super::diffusion::cosine_similarity;

#[derive(Clone, Copy, Debug)]
/// Tunable coefficients governing the reward calculation.
pub struct RewardConfig {
    /// Weight applied to logical correctness of spike patterns.
    pub alpha: f32,
    /// Penalty factor applied to activation sparsity.
    pub beta: f32,
    /// Scaling factor for temporal consistency between diffusion steps.
    pub gamma: f32,
    /// Activation level treated as "on" when measuring sparsity.
    pub activation_threshold: f32,
}

#[derive(Clone, Copy, Debug)]
/// Breakdown of reward components for diagnostics and tests.
pub struct RewardBreakdown {
    /// Aggregated scalar reward.
    pub reward: f32,
    /// Contribution from expected versus observed spikes.
    pub logic: f32,
    /// Fraction of active neurons relative to the configured threshold.
    pub sparsity: f32,
    /// Cosine similarity between consecutive diffusion states.
    pub consistency: f32,
}

/// Stateful reward calculator that remembers the previous activation snapshot.
pub struct RewardCalculator {
    config: RewardConfig,
    previous_state: Option<Vec<f32>>,
}

impl RewardCalculator {
    /// Creates a calculator using the supplied [`RewardConfig`].
    pub fn new(config: RewardConfig) -> Self {
        Self {
            config,
            previous_state: None,
        }
    }

    /// Evaluates reward components for a single reasoning step.
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

/// Computes a simple accuracy-style score over spike sets.
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

#[cfg(test)]
mod tests {
    use super::logic_score;

    #[test]
    fn empty_expected_penalises_spurious_spikes() {
        assert_eq!(logic_score(&[], &[1, 2]), -1.0);
    }

    #[test]
    fn perfect_match_scores_one() {
        assert_eq!(logic_score(&[1, 2], &[1, 2]), 1.0);
    }
}
