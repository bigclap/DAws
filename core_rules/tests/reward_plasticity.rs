use core_graph::test_helpers;
use core_rules::reward::{RewardCalculator, RewardConfig};

#[test]
fn reward_modulated_stdp_increases_weight_on_positive_reward() {
    let mut network = test_helpers::simple_pair();
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
