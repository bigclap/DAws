//! Lightweight profiling helpers for tracking network activity during runs.

use crate::{Network, StepReport};

/// Trait implemented by observers that wish to inspect step reports.
pub trait StepObserver {
    /// Invoked after each scheduled network step.
    fn on_step(&mut self, step: usize, network: &Network, report: &StepReport);
}

/// Configuration controlling derived metrics recorded by the profiler.
#[derive(Clone, Copy, Debug)]
pub struct ProfilerConfig {
    /// Activation threshold used when computing active ratios.
    pub activation_threshold: f32,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            activation_threshold: 0.2,
        }
    }
}

/// Snapshot of a single step containing both spikes and aggregate metrics.
#[derive(Clone, Debug, PartialEq)]
pub struct StepSnapshot {
    pub step: usize,
    pub spikes: Vec<usize>,
    pub modulatory_spikes: Vec<usize>,
    pub energy: f32,
    pub active_ratio: f32,
}

/// Aggregate statistics derived from the recorded snapshots.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProfileSummary {
    pub steps: usize,
    pub average_energy: f32,
    pub average_active_ratio: f32,
}

/// Profiler implementation that collects per-step snapshots.
#[derive(Debug, Default)]
pub struct NetworkProfiler {
    config: ProfilerConfig,
    snapshots: Vec<StepSnapshot>,
}

impl NetworkProfiler {
    /// Creates a profiler that records step snapshots.
    pub fn new(config: ProfilerConfig) -> Self {
        Self {
            config,
            snapshots: Vec::new(),
        }
    }

    /// Clears previously recorded snapshots, making the profiler reusable.
    pub fn reset(&mut self) {
        self.snapshots.clear();
    }

    /// Returns an immutable view over the recorded snapshots.
    pub fn snapshots(&self) -> &[StepSnapshot] {
        &self.snapshots
    }

    /// Computes summary statistics across all snapshots.
    pub fn summary(&self) -> ProfileSummary {
        if self.snapshots.is_empty() {
            return ProfileSummary {
                steps: 0,
                average_energy: 0.0,
                average_active_ratio: 0.0,
            };
        }
        let mut energy_sum = 0.0;
        let mut active_sum = 0.0;
        for snapshot in &self.snapshots {
            energy_sum += snapshot.energy;
            active_sum += snapshot.active_ratio;
        }
        let count = self.snapshots.len() as f32;
        ProfileSummary {
            steps: self.snapshots.len(),
            average_energy: energy_sum / count,
            average_active_ratio: active_sum / count,
        }
    }
}

impl StepObserver for NetworkProfiler {
    fn on_step(&mut self, step: usize, network: &Network, report: &StepReport) {
        let snapshot = StepSnapshot {
            step,
            spikes: report.spikes.clone(),
            modulatory_spikes: report.modulatory_spikes.clone(),
            energy: network.energy(),
            active_ratio: network.active_ratio(self.config.activation_threshold),
        };
        self.snapshots.push(snapshot);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembly::{ConnectionParams, GraphBuilder, NodeParams};

    #[test]
    fn profiler_records_snapshots_and_summary() {
        let mut builder = GraphBuilder::new();
        let input = builder.add_input_node(NodeParams::default());
        let output = builder.add_node(NodeParams::default());
        builder.add_connection(ConnectionParams::new(
            input, output, 1.0, 1.0, 0, 1.0, 1.0, 5.0, 5.0,
        ));
        let mut network = builder.build().expect("valid network");

        let mut profiler = NetworkProfiler::new(ProfilerConfig::default());
        let report = network.step(0);
        profiler.on_step(0, &network, &report);

        assert_eq!(profiler.snapshots().len(), 1);
        let summary = profiler.summary();
        assert_eq!(summary.steps, 1);
        assert!(summary.average_energy >= 0.0);
    }

    #[test]
    fn summary_is_zero_for_empty_profiler() {
        let profiler = NetworkProfiler::new(ProfilerConfig::default());
        let summary = profiler.summary();
        assert_eq!(summary.steps, 0);
        assert_eq!(summary.average_energy, 0.0);
        assert_eq!(summary.average_active_ratio, 0.0);
    }
}
