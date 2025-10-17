use approx::assert_relative_eq;
use DAws::signal::test_helpers;

#[test]
fn modulatory_nodes_temporarily_raise_thresholds() {
    let mut network = test_helpers::two_node_modulatory();
    network.inject(&[(0, 1.0)]);
    let report = network.step(0);
    assert!(report.modulatory_spikes.contains(&0));
    let target = network.node(1);
    assert!(target.effective_threshold() > target.base_threshold + target.adaptation);

    let _ = network.step(1);
    let target = network.node(1);
    assert_relative_eq!(
        target.effective_threshold(),
        target.base_threshold + target.adaptation,
        epsilon = 1e-6
    );
}
