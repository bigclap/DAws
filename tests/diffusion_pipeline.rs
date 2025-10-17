use DAws::diffusion::{DiffusionConfig, DiffusionLoop};
use DAws::io::{BinaryDecoder, TableEncoder};
use DAws::signal::{build_xor_network, Network};
use rstest::{fixture, rstest};

type XorPipeline = (
    Network,
    TableEncoder,
    BinaryDecoder,
    usize,
    DiffusionLoop,
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
    (network, encoder, decoder, output_node, diffusion)
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
    let (mut network, encoder, decoder, output_node, mut diffusion) = ctx;
    let embedding = encoder.encode(input);
    network.reset_state();
    network.inject_embedding(&embedding);
    let _ = network.step(0);
    let _ = network.step(1);
    let _ = network.step(2);

    let state = diffusion.run(&mut network).state;
    let output = decoder.decode(state[output_node]);
    assert_eq!(output, expected, "failed for input {input}");

    let active_ratio = network.active_ratio(0.2);
    assert!(active_ratio <= 0.6, "active ratio {active_ratio}");
    assert!(diffusion.last_iterations() <= 10);
    assert!(diffusion.last_similarity() > 0.95);
}
