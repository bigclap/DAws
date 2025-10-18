use core_graph::assembly::GraphBuilder;
use core_graph::{InhibitoryPoolConfig, NodeParams, NodeType, RegionalDetectorConfig};

fn excitatory_params(threshold: f32) -> NodeParams {
    NodeParams {
        node_type: NodeType::Excitatory,
        base_threshold: threshold,
        lambda_v: 1.0,
        lambda_h: 1.0,
        activation_decay: 1.0,
        divisive_beta: 0.0,
        kappa: 0.0,
        modulation_decay: 1.0,
        ..NodeParams::default()
    }
}

#[test]
fn inhibitory_pool_promotes_winner_take_all() {
    let mut builder = GraphBuilder::new();
    let a = builder.add_input_node(excitatory_params(0.3));
    let b = builder.add_input_node(excitatory_params(0.3));
    let c = builder.add_input_node(excitatory_params(0.3));

    builder.add_inhibitory_pool(InhibitoryPoolConfig::new(vec![a, b, c], 1.2));

    let mut network = builder.build().expect("valid network");

    network.inject(&[(a, 1.0), (b, 0.8), (c, 0.6)]);
    let report = network.step(0);

    assert_eq!(report.spikes, vec![a]);
    assert!(report.modulatory_spikes.is_empty());
}

#[test]
fn regional_detector_tracks_activity_and_refresh_window() {
    let mut builder = GraphBuilder::new();
    let a = builder.add_input_node(excitatory_params(0.4));
    let b = builder.add_input_node(excitatory_params(0.4));

    let detector = RegionalDetectorConfig {
        label: "region-A".to_string(),
        activation_threshold: 0.5,
        refresh_interval: 2,
    };

    builder.add_inhibitory_pool(
        InhibitoryPoolConfig::new(vec![a, b], 0.8).with_detector(detector.clone()),
    );

    let mut network = builder.build().expect("valid network");

    network.inject(&[(a, 0.9)]);
    let _ = network.step(0);

    let state = network
        .regional_detector_state(&detector.label)
        .expect("detector configured");
    assert_eq!(state.winner, Some(a));
    assert!(!state.needs_refresh);

    for step in 1..=2 {
        let _ = network.step(step);
    }

    let state = network
        .regional_detector_state(&detector.label)
        .expect("detector configured");
    assert_eq!(state.last_active_step, Some(0));
    assert!(state.needs_refresh);
    assert!(state.stale_steps >= detector.refresh_interval);
}

#[test]
fn builder_rejects_pools_with_unknown_members() {
    let mut builder = GraphBuilder::new();
    builder.add_inhibitory_pool(InhibitoryPoolConfig::new(vec![0], 1.0));
    let error = builder.build().err().expect("missing node error");
    assert_eq!(
        error,
        core_graph::AssemblyError::MissingPoolNode { node_id: 0 }
    );
}
