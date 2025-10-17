//! Diffusion loop that stabilises the network state vector between graph steps.

use core_graph::Network;

#[derive(Clone, Copy, Debug)]
/// Parameters controlling the semantic diffusion loop.
pub struct DiffusionConfig {
    /// Interpolation factor toward the consensus state on each iteration.
    pub alpha: f32,
    /// Convergence tolerance measured as `(1 - cosine_similarity)`.
    pub tolerance: f32,
    /// Hard cap on the number of refinement steps.
    pub max_iters: usize,
    /// Amplitude of deterministic sinusoidal noise injected per dimension.
    pub noise: f32,
}

#[derive(Debug, Clone)]
/// Final state returned by a completed diffusion run.
pub struct DiffusionOutcome {
    /// Copy of the network activation vector after convergence or timeout.
    pub state: Vec<f32>,
}

/// Performs deterministic diffusion updates on a [`Network`] activation vector.
pub struct DiffusionLoop {
    config: DiffusionConfig,
    last_similarity: f32,
    last_iterations: usize,
}

impl DiffusionLoop {
    /// Creates a diffusion loop with the supplied [`DiffusionConfig`].
    pub fn new(config: DiffusionConfig) -> Self {
        Self {
            config,
            last_similarity: 0.0,
            last_iterations: 0,
        }
    }

    /// Runs the diffusion iterations until convergence or the iteration budget is
    /// exhausted, updating the provided [`Network`] in place and returning the
    /// final state vector.
    pub fn run(&mut self, network: &mut Network) -> DiffusionOutcome {
        let mut current = network.state_vector();
        if current.is_empty() {
            self.last_similarity = 1.0;
            self.last_iterations = 0;
            return DiffusionOutcome { state: current };
        }

        let mut last_similarity = 1.0;
        for iter in 0..self.config.max_iters {
            let consensus = network.consensus_state();
            let mut next = Vec::with_capacity(consensus.len());
            for (idx, (value, consensus_value)) in current.iter().zip(consensus.iter()).enumerate()
            {
                let delta = self.config.alpha * (consensus_value - value);
                let noise = deterministic_noise(idx, iter, self.config.noise);
                next.push((value + delta + noise).clamp(0.0, 1.0));
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

    /// Returns the cosine similarity between the last two diffusion states.
    pub fn last_similarity(&self) -> f32 {
        self.last_similarity
    }

    /// Returns the number of iterations consumed by the last diffusion run.
    pub fn last_iterations(&self) -> usize {
        self.last_iterations
    }
}

/// Computes cosine similarity between two activation vectors.
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

/// Deterministic pseudo-noise source used by [`DiffusionLoop::run`].
fn deterministic_noise(index: usize, iteration: usize, amplitude: f32) -> f32 {
    if amplitude == 0.0 {
        return 0.0;
    }
    let phase = (index as f32 * 0.618_033_9 + iteration as f32 * 0.414_213_56).sin();
    phase * amplitude
}

#[cfg(test)]
mod tests {
    use super::{cosine_similarity, deterministic_noise};

    #[test]
    fn cosine_similarity_returns_one_for_zero_vectors() {
        let a = [0.0f32; 3];
        let b = [0.0f32; 3];
        assert_eq!(cosine_similarity(&a, &b), 1.0);
    }

    #[test]
    fn deterministic_noise_zero_amplitude_is_silent() {
        assert_eq!(deterministic_noise(3, 7, 0.0), 0.0);
    }
}
