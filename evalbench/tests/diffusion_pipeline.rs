use core_graph::{Network, NetworkProfiler, ProfilerConfig};
use core_rules::diffusion::{DiffusionConfig, DiffusionLoop};
use core_rules::scheduler::{ReasoningScheduler, SchedulerConfig};
use evalbench::build_xor_network;
use model_dec::BinaryDecoder;
use model_enc::TableEncoder;
use rstest::{fixture, rstest};

type XorPipeline = (
    Network,
    TableEncoder,
    BinaryDecoder,
    usize,
    DiffusionLoop,
    ReasoningScheduler,
    ProfilerConfig,
);

#[fixture]
fn xor_pipeline() -> XorPipeline {
    let (network, encoder, decoder, output_node) = build_xor_network();
    let diffusion = DiffusionLoop::new(DiffusionConfig {
        alpha: 0.5,
        tolerance: 1e-3,
        max_iters: 10,
        noise: 0.0,
    });
    let scheduler = ReasoningScheduler::new(SchedulerConfig { settle_steps: 3 });
    let profiler_cfg = ProfilerConfig {
        activation_threshold: 0.2,
    };
    (
        network,
        encoder,
        decoder,
        output_node,
        diffusion,
        scheduler,
        profiler_cfg,
    )
}

#[rstest]
#[case("0 0", "0")]
#[case("0 1", "1")]
#[case("1 0", "1")]
#[case("1 1", "0")]
fn xor_reasoning_pipeline_produces_expected_outputs(
    #[case] input: &str,
    #[case] expected: &str,
    #[from(xor_pipeline)] ctx: XorPipeline,
) {
    let (mut network, encoder, decoder, output_node, mut diffusion, scheduler, profiler_cfg) = ctx;
    let embedding = encoder.encode(input);
    let mut profiler = NetworkProfiler::new(profiler_cfg);
    let outcome = scheduler.run_case(
        &mut network,
        &embedding,
        &mut diffusion,
        Some(&mut profiler),
    );
    let output = decoder.decode(outcome.state[output_node].clamp(0.0, 1.0));
    assert_eq!(output, expected, "failed for input {input}");

    let summary = profiler.summary();
    assert!(summary.average_active_ratio <= 0.6);
    assert!(outcome.iterations <= 10);
    assert!(outcome.similarity > 0.95);
}
