//! CLI entry point that showcases the reasoning pipeline on an XOR task.

mod telemetry;

use core_graph::{NetworkProfiler, ProfilerConfig};
use core_rules::diffusion::{AnnealingSchedule, DiffusionConfig, DiffusionLoop, EntropyPolicy};
use core_rules::scheduler::{ReasoningScheduler, SchedulerConfig};
use evalbench::build_xor_network;
use telemetry::{init_telemetry, write_profile};

const PROFILE_OUTPUT: &str = "cpu_profile.svg";

/// Runs the XOR demonstration and prints inference diagnostics.
fn main() {
    let profiler_guard = init_telemetry();
    let (mut network, encoder, decoder, output_node) = build_xor_network();
    let mut diffusion = DiffusionLoop::new(DiffusionConfig {
        alpha_schedule: AnnealingSchedule::constant(0.5),
        sigma_schedule: AnnealingSchedule::constant(0.0),
        tolerance: 1e-3,
        jt_tolerance: 5e-4,
        stability_tolerance: 5e-4,
        stability_window: 2,
        max_energy_increase: usize::MAX,
        max_iters: 20,
        entropy_policy: EntropyPolicy::default(),
        fact_recruitment: None,
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
        let diagnostics = diffusion.diagnostics().clone();
        tracing::info!(
            case = input,
            output = %output,
            iterations = outcome.iterations,
            similarity = outcome.similarity,
            avg_energy = summary.average_energy,
            avg_active_nodes = summary.average_active_nodes,
            avg_active_edges = summary.average_active_edges,
            avg_fragmentation = summary.average_fragmentation,
            avg_iteration_ms = diagnostics.average_iteration_ms,
            "inference result",
        );
        println!(
            "{input} -> {output} (iters={}, similarity={:.3}, avg_energy={:.3}, avg_active_nodes={:.1}, avg_active_edges={:.1}, avg_fragmentation={:.3}, avg_iter_ms={:.3})",
            outcome.iterations,
            outcome.similarity,
            summary.average_energy,
            summary.average_active_nodes,
            summary.average_active_edges,
            summary.average_fragmentation,
            diagnostics.average_iteration_ms,
        );
    }
    println!("Energy footprint: {:.3}", network.energy());
    if let Some(guard) = profiler_guard {
        write_profile(guard, PROFILE_OUTPUT);
        println!("CPU profile written to {PROFILE_OUTPUT}");
    }
}
