use super::*;
use core_graph::{ConnectionParams, NodeParams, NodeType, assembly::GraphBuilder};
use retriever::{Retriever, RetrieverConfig};

#[test]
fn schedule_and_similarity_helpers_behave() {
    let schedule = AnnealingSchedule::new(vec![0.5, 0.25]);
    assert_eq!(schedule.value_at(2), 0.25);
    let a = [0.0f32; 3];
    assert_eq!(cosine_similarity(&a, &a), 1.0);
    assert_eq!(deterministic_noise(3, 7, 0.0), 0.0);
}

#[test]
fn diffusion_applies_entropy_scaled_gain() {
    let mut builder = GraphBuilder::new();
    let driver = builder.add_node(NodeParams {
        node_type: NodeType::Excitatory,
        ..NodeParams::default()
    });
    let target = builder.add_node(NodeParams::default());
    builder.add_connection(ConnectionParams::new(
        driver, target, 1.0, 1.0, 0, 1.0, 1.0, 5.0, 5.0,
    ));
    let mut network = builder.build().expect("valid network");
    network.set_state(&[1.0, 0.0]);

    let mut diffusion = DiffusionLoop::new(DiffusionConfig {
        alpha_schedule: AnnealingSchedule::constant(0.5),
        sigma_schedule: AnnealingSchedule::constant(0.0),
        tolerance: 0.0,
        jt_tolerance: 1e-6,
        stability_tolerance: 1e-3,
        stability_window: 1,
        max_energy_increase: 0,
        max_iters: 1,
        entropy_policy: EntropyPolicy {
            low_threshold: 0.4,
            high_threshold: 0.95,
            boost: 1.0,
            dampening: 0.0,
        },
        fact_recruitment: None,
    });
    let outcome = diffusion.run(&mut network);
    assert!(diffusion.last_iterations() >= 1);
    assert!((outcome.state[1] - 0.5).abs() < 1e-6);
}

#[test]
fn fact_recruitment_selects_high_affinity_fact() {
    let config = RetrieverConfig {
        dimension: 2,
        value_dimension: 2,
        max_elements: 8,
        max_layers: 4,
        max_connections: 4,
        ef_construction: 16,
        ef_search: 8,
        top_k: 1,
    };
    let retriever = Retriever::new(config).expect("valid retriever");
    let recruitment =
        FactRecruitment::new(retriever, vec![vec![1.0, 0.0], vec![0.0, 1.0]], 0.6, 0.5)
            .expect("facts ingested");
    let contribution = recruitment.recruit(&[1.0, 0.0]).expect("fact retrieved");
    assert!((contribution[0] - 0.6).abs() < 1e-6);
    assert!(contribution[1] <= 1e-6);
}

#[test]
fn diffusion_reports_convergence_diagnostics() {
    let mut builder = GraphBuilder::new();
    let driver = builder.add_node(NodeParams {
        node_type: NodeType::Excitatory,
        ..NodeParams::default()
    });
    let target = builder.add_node(NodeParams::default());
    builder.add_connection(ConnectionParams::new(
        driver, target, 1.0, 1.0, 0, 1.0, 1.0, 5.0, 5.0,
    ));
    let mut network = builder.build().expect("valid network");
    network.set_state(&[1.0, 0.0]);

    let mut diffusion = DiffusionLoop::new(DiffusionConfig {
        alpha_schedule: AnnealingSchedule::constant(0.4),
        sigma_schedule: AnnealingSchedule::constant(0.0),
        tolerance: 1e-3,
        jt_tolerance: 1e-3,
        stability_tolerance: 1e-3,
        stability_window: 2,
        max_energy_increase: usize::MAX,
        max_iters: 8,
        entropy_policy: EntropyPolicy::default(),
        fact_recruitment: None,
    });

    let outcome = diffusion.run(&mut network);
    let diagnostics = outcome.diagnostics;
    assert!(diagnostics.iterations <= 8);
    assert!(diagnostics.similarity <= 1.0 + 1e-6);
    assert!(diagnostics.jt >= 0.0);
    assert!(diagnostics.stability_streak <= 2);
    assert!(diagnostics.energy >= 0.0);
    assert!(!diagnostics.energy_monotonic || diagnostics.iterations <= 1);
}
