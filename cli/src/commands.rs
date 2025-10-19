use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use core_graph::{NetworkProfiler, ProfileSummary, ProfilerConfig};
use core_rules::diffusion::{AnnealingSchedule, DiffusionConfig, DiffusionLoop, EntropyPolicy};
use core_rules::scheduler::{ReasoningScheduler, SchedulerConfig};
use evalbench::build_xor_network;
use tracing::{info, warn};
use trainer::{MmapDataset, OfflineDecoderTrainer, ValidationRecord, ValidationReport};

use crate::config::{EvalSettings, InferSettings, ProfileSettings, TrainSettings};
use crate::telemetry::{init_telemetry, write_profile};

pub fn run_train(config_path: Option<PathBuf>) -> Result<()> {
    let settings = crate::config::load_settings::<TrainSettings>("train", config_path)?;
    let profiler_guard = init_telemetry();

    if let Some(dataset) = &settings.dataset {
        info!(path = %dataset.display(), "training dataset located");
    } else {
        info!("no dataset supplied, using synthetic validation records");
    }
    info!(?settings.trainer, "offline trainer configuration");

    let records = load_validation_records(settings.dataset.as_deref())?;
    let trainer = OfflineDecoderTrainer::new(settings.trainer.clone());
    let report = trainer.validate(records)?;
    let report_text = render_report(&report);
    println!("Training summary:\n{report_text}");

    let checkpoint_body = format!("# Offline decoder checkpoint summary\n{report_text}");
    write_text_file(&settings.checkpoint, &checkpoint_body)?;
    println!(
        "Checkpoint summary written to {}",
        settings.checkpoint.display()
    );

    if let Some(guard) = profiler_guard {
        if let Some(profile_path) = settings.profile_output {
            ensure_parent(&profile_path)?;
            write_profile(guard, &profile_path);
            println!("CPU profile written to {}", profile_path.display());
        }
    }

    Ok(())
}

pub fn run_eval(config_path: Option<PathBuf>) -> Result<()> {
    let settings = crate::config::load_settings::<EvalSettings>("eval", config_path)?;
    let profiler_guard = init_telemetry();

    if settings.checkpoint.exists() {
        info!(checkpoint = %settings.checkpoint.display(), "evaluating checkpoint");
    } else {
        warn!(
            checkpoint = %settings.checkpoint.display(),
            "checkpoint not found, continuing with evaluation"
        );
    }

    let records = load_validation_records(settings.dataset.as_deref())?;
    let trainer = OfflineDecoderTrainer::new(settings.trainer.clone());
    let report = trainer.validate(records)?;
    let report_text = render_report(&report);
    println!("Evaluation summary:\n{report_text}");

    let report_body = format!("# Evaluation summary\n{report_text}");
    write_text_file(&settings.report, &report_body)?;
    println!("Evaluation report written to {}", settings.report.display());

    if let Some(guard) = profiler_guard {
        if let Some(profile_path) = settings.profile_output {
            ensure_parent(&profile_path)?;
            write_profile(guard, &profile_path);
            println!("CPU profile written to {}", profile_path.display());
        }
    }

    Ok(())
}

pub fn run_infer(config_path: Option<PathBuf>) -> Result<()> {
    let settings = crate::config::load_settings::<InferSettings>("infer", config_path)?;
    let profiler_guard = init_telemetry();

    if let Some(checkpoint) = &settings.checkpoint {
        println!("Using checkpoint hint: {}", checkpoint.display());
    }

    let summary = run_xor_inference(&settings.inputs);
    print_inference_summary(&summary);

    if let Some(guard) = profiler_guard {
        if let Some(profile_path) = settings.profile_output {
            ensure_parent(&profile_path)?;
            write_profile(guard, &profile_path);
            println!("CPU profile written to {}", profile_path.display());
        }
    }

    Ok(())
}

pub fn run_profile(config_path: Option<PathBuf>) -> Result<()> {
    let settings = crate::config::load_settings::<ProfileSettings>("profile", config_path)?;
    let profiler_guard = init_telemetry();

    let summary = run_xor_inference(&settings.inputs);
    print_inference_summary(&summary);

    if let Some(guard) = profiler_guard {
        ensure_parent(&settings.profile_output)?;
        write_profile(guard, &settings.profile_output);
        println!(
            "CPU profile written to {}",
            settings.profile_output.display()
        );
    }

    Ok(())
}

fn run_xor_inference(cases: &[String]) -> InferenceSummary {
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

    let mut results = Vec::new();

    for case in cases {
        let embedding = encoder.encode(case);
        profiler.reset();
        let outcome = scheduler.run_case(
            &mut network,
            &embedding,
            &mut diffusion,
            Some(&mut profiler),
        );
        let activation = outcome.state[output_node].clamp(0.0, 1.0);
        let output = decoder.decode(activation);
        let profile = profiler.summary();
        let diagnostics = diffusion.diagnostics().clone();

        info!(
            case = case.as_str(),
            output = %output,
            iterations = outcome.iterations,
            similarity = outcome.similarity,
            avg_energy = profile.average_energy,
            avg_active_nodes = profile.average_active_nodes,
            avg_active_edges = profile.average_active_edges,
            avg_fragmentation = profile.average_fragmentation,
            avg_iteration_ms = diagnostics.average_iteration_ms,
            stop_reason = ?diagnostics.stop_reason,
            "inference result",
        );

        results.push(CaseReport {
            case: case.to_string(),
            output,
            iterations: outcome.iterations,
            similarity: outcome.similarity,
            profile,
            average_iteration_ms: diagnostics.average_iteration_ms,
        });
    }

    InferenceSummary {
        energy: network.energy(),
        cases: results,
    }
}

fn print_inference_summary(summary: &InferenceSummary) {
    println!("PoC XOR reasoning demonstration:");
    for case in &summary.cases {
        println!(
            "{case_label} -> {output} (iters={iters}, similarity={similarity:.3}, avg_energy={energy:.3}, \\navg_active_nodes={active_nodes:.1}, avg_active_edges={active_edges:.1}, avg_fragmentation={fragmentation:.3}, avg_iter_ms={iter_ms:.3})",
            case_label = case.case,
            output = case.output,
            iters = case.iterations,
            similarity = case.similarity,
            energy = case.profile.average_energy,
            active_nodes = case.profile.average_active_nodes,
            active_edges = case.profile.average_active_edges,
            fragmentation = case.profile.average_fragmentation,
            iter_ms = case.average_iteration_ms,
        );
    }
    println!("Energy footprint: {:.3}", summary.energy);
}

fn load_validation_records(dataset: Option<&Path>) -> Result<Vec<ValidationRecord>> {
    match dataset {
        Some(path) => {
            let mmap = MmapDataset::open(path)
                .with_context(|| format!("failed to open dataset at {}", path.display()))?;
            let mut records = Vec::new();
            for sample in mmap.iter() {
                let sample = sample.with_context(|| {
                    format!("failed to parse dataset record from {}", path.display())
                })?;
                let token_count = sample.target_tokens.len().max(1);
                let log_probs = vec![-0.1; token_count];
                let mut retrieval_candidates =
                    Vec::with_capacity(sample.retrieval_candidates.len() + 1);
                retrieval_candidates.push(sample.target_embedding.clone());
                retrieval_candidates.extend(sample.retrieval_candidates.clone());
                let generated_tokens = sample.target_tokens.clone();
                records.push(ValidationRecord {
                    predicted_embedding: sample.target_embedding.clone(),
                    target_embedding: sample.target_embedding,
                    retrieval_candidates,
                    generated_tokens,
                    target_tokens: sample.target_tokens,
                    log_probs,
                });
            }
            if records.is_empty() {
                bail!("dataset at {} did not contain any samples", path.display());
            }
            Ok(records)
        }
        None => Ok(vec![ValidationRecord {
            predicted_embedding: vec![1.0, 0.0],
            target_embedding: vec![1.0, 0.0],
            retrieval_candidates: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            generated_tokens: vec!["1".into()],
            target_tokens: vec!["1".into()],
            log_probs: vec![-0.1],
        }]),
    }
}

fn render_report(report: &ValidationReport) -> String {
    let mut lines = vec![
        format!("cosine_at_median = {:.6}", report.cosine_at_median),
        format!("distinct_n = {:.6}", report.distinct_n),
        format!("ppl_surrogate = {:.6}", report.ppl_surrogate),
    ];
    for (k, value) in &report.retrieval_rank_at_k {
        lines.push(format!("retrieval@{} = {:.6}", k, value));
    }
    lines.join("\n")
}

fn write_text_file(path: &Path, contents: &str) -> Result<()> {
    ensure_parent(path)?;
    let mut body = contents.to_string();
    if !body.ends_with('\n') {
        body.push('\n');
    }
    fs::write(path, body).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

fn ensure_parent(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create directory {}", parent.display()))?;
        }
    }
    Ok(())
}

struct CaseReport {
    case: String,
    output: String,
    iterations: usize,
    similarity: f32,
    profile: ProfileSummary,
    average_iteration_ms: f32,
}

struct InferenceSummary {
    energy: f32,
    cases: Vec<CaseReport>,
}
