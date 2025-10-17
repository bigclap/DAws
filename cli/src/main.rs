//! CLI entry point that showcases the reasoning pipeline on an XOR task.

use core_rules::diffusion::{DiffusionConfig, DiffusionLoop};
use evalbench::build_xor_network;

/// Runs the XOR demonstration and prints inference diagnostics.
fn main() {
    let (mut network, encoder, decoder, output_node) = build_xor_network();
    let mut diffusion = DiffusionLoop::new(DiffusionConfig {
        alpha: 0.5,
        tolerance: 1e-3,
        max_iters: 20,
        noise: 0.0,
    });

    let cases = ["0 0", "0 1", "1 0", "1 1"];

    println!("PoC XOR reasoning demonstration:");
    for input in cases {
        let embedding = encoder.encode(input);
        network.reset_state();
        network.inject_embedding(&embedding);
        let _ = network.step(0);
        let _ = network.step(1);
        let _ = network.step(2);
        let state = diffusion.run(&mut network).state;
        let output = decoder.decode(state[output_node].clamp(0.0, 1.0));
        println!(
            "{input} -> {output} (iters={}, similarity={:.3})",
            diffusion.last_iterations(),
            diffusion.last_similarity()
        );
    }
    println!("Energy footprint: {:.3}", network.energy());
}
