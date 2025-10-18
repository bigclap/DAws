//! CLI entry point that showcases the reasoning pipeline on an XOR task.

use core_graph::{NetworkProfiler, ProfilerConfig};
use core_rules::diffusion::{DiffusionConfig, DiffusionLoop};
use core_rules::scheduler::{ReasoningScheduler, SchedulerConfig};
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
    let scheduler = ReasoningScheduler::new(SchedulerConfig { settle_steps: 3 });
    let mut profiler = NetworkProfiler::new(ProfilerConfig {
        activation_threshold: 0.2,
    });

    let cases = ["0 0", "0 1", "1 0", "1 1"];

    println!("PoC XOR reasoning demonstration:");
    for input in cases {
        let embedding = encoder.encode(input);
        profiler.reset();
        let outcome = scheduler.run_case(
            &mut network,
            &embedding,
            &mut diffusion,
            Some(&mut profiler),
        );
        let output = decoder.decode(outcome.state[output_node].clamp(0.0, 1.0));
        let summary = profiler.summary();
        println!(
            "{input} -> {output} (iters={}, similarity={:.3}, avg_energy={:.3})",
            outcome.iterations, outcome.similarity, summary.average_energy
        );
    }
    println!("Energy footprint: {:.3}", network.energy());
}
