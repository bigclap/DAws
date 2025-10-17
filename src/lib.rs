//! Core library exposing the event-driven network, diffusion loop and reward shaping utilities.

pub mod diffusion;
pub mod io;
pub mod reward;
pub mod signal;

#[cfg(test)]
mod tests {
    use super::diffusion::{DiffusionConfig, DiffusionLoop};
    use super::reward::{RewardCalculator, RewardConfig};
    use super::signal::build_xor_network;

    #[test]
    fn xor_reasoning_pipeline_produces_expected_outputs() {
        let (mut network, encoder, decoder, output_node) = build_xor_network();
        let mut diffusion = DiffusionLoop::new(DiffusionConfig {
            alpha: 0.5,
            tolerance: 1e-3,
            max_iters: 10,
            noise: 0.0,
        });

        let cases = [("0 0", "0"), ("0 1", "1"), ("1 0", "1"), ("1 1", "0")];

        for (input, expected) in cases {
            let embedding = encoder.encode(input);
            network.reset_state();
            network.inject_embedding(&embedding);
            let _ = network.step(0);
            let _report1 = network.step(1);
            let _report2 = network.step(2);

            let state = diffusion.run(&mut network).state;
            let output = decoder.decode(state[output_node]);
            assert_eq!(output, expected, "failed for input {input}");

            // sparsity target measured on activations
            let active_ratio = network.active_ratio(0.2);
            assert!(active_ratio <= 0.6, "active ratio {active_ratio}");

            // check iteration bound
            assert!(diffusion.last_iterations() <= 10);

            // ensure convergence metric high
            assert!(diffusion.last_similarity() > 0.95);
        }
    }

    #[test]
    fn modulatory_nodes_temporarily_raise_thresholds() {
        let mut network = super::signal::test_helpers::two_node_modulatory();
        network.inject(&[(0, 1.0)]);
        let report = network.step(0);
        assert!(report.modulatory_spikes.contains(&0));
        let target = network.node(1);
        assert!(target.effective_threshold() > target.base_threshold + target.adaptation);
        // modulation should decay after another step without activity
        let _ = network.step(1);
        let target = network.node(1);
        assert!(
            (target.effective_threshold() - (target.base_threshold + target.adaptation)).abs()
                < 1e-6
        );
    }

    #[test]
    fn reward_modulated_stdp_increases_weight_on_positive_reward() {
        let mut network = super::signal::test_helpers::simple_pair();
        network.inject(&[(0, 1.0)]);
        let _ = network.step(0);
        network.inject(&[(1, 1.0)]);
        let report = network.step(1);
        assert!(report.spikes.contains(&1));

        let mut calculator = RewardCalculator::new(RewardConfig {
            alpha: 1.0,
            beta: 0.0,
            gamma: 0.0,
            activation_threshold: 0.1,
        });
        let reward = calculator.evaluate(&network, &[1], &report.spikes).reward;
        assert!(reward > 0.0);
        let before = network.connection_weight(0);
        network.apply_reward(reward, 0.05);
        let after = network.connection_weight(0);
        assert!(after > before);
    }
}
