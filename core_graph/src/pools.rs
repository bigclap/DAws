use crate::Node;

/// Declarative description of an inhibitory pool.
#[derive(Clone, Debug)]
pub struct InhibitoryPoolConfig {
    /// Nodes governed by the pool.
    pub members: Vec<usize>,
    /// Amount of inhibition applied to non-winning members each step.
    pub inhibition_gain: f32,
    /// Optional detector capturing regional activity metadata.
    pub detector: Option<RegionalDetectorConfig>,
}

impl InhibitoryPoolConfig {
    /// Creates a new pool configuration targeting the supplied members.
    pub fn new(members: Vec<usize>, inhibition_gain: f32) -> Self {
        Self {
            members,
            inhibition_gain,
            detector: None,
        }
    }

    /// Attaches a detector configuration to the pool.
    pub fn with_detector(mut self, detector: RegionalDetectorConfig) -> Self {
        self.detector = Some(detector);
        self
    }
}

/// Configuration of a detector tracking regional activity.
#[derive(Clone, Debug)]
pub struct RegionalDetectorConfig {
    /// Identifier used when querying detector state.
    pub label: String,
    /// Minimum activation required to treat the region as active.
    pub activation_threshold: f32,
    /// Number of consecutive inactive steps before signalling refresh.
    pub refresh_interval: usize,
}

/// Snapshot describing the state of a configured regional detector.
#[derive(Clone, Debug, PartialEq)]
pub struct RegionalDetectorState {
    pub label: String,
    pub winner: Option<usize>,
    pub peak_activation: f32,
    pub last_active_step: Option<usize>,
    pub stale_steps: usize,
    pub needs_refresh: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct InhibitoryPoolRuntime {
    members: Vec<usize>,
    inhibition_gain: f32,
    detector_index: Option<usize>,
}

impl InhibitoryPoolRuntime {
    pub(crate) fn new(
        members: Vec<usize>,
        inhibition_gain: f32,
        detector_index: Option<usize>,
    ) -> Self {
        Self {
            members,
            inhibition_gain,
            detector_index,
        }
    }

    pub(crate) fn members(&self) -> &[usize] {
        &self.members
    }

    pub(crate) fn inhibition_gain(&self) -> f32 {
        self.inhibition_gain
    }

    pub(crate) fn detector_index(&self) -> Option<usize> {
        self.detector_index
    }

    pub(crate) fn determine_winner(&self, nodes: &[Node]) -> (Option<usize>, f32) {
        let mut best: Option<(usize, f32)> = None;
        for &node_id in &self.members {
            let Some(node) = nodes.get(node_id) else {
                continue;
            };
            let value = node.potential;
            if !value.is_finite() {
                continue;
            }
            best = match best {
                None => Some((node_id, value)),
                Some((_, current)) if value > current => Some((node_id, value)),
                Some(existing) => Some(existing),
            };
        }
        if let Some((id, value)) = best {
            if value > 0.0 {
                (Some(id), value)
            } else {
                (None, 0.0)
            }
        } else {
            (None, 0.0)
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct RegionalDetectorRuntime {
    label: String,
    activation_threshold: f32,
    refresh_interval: usize,
    last_active_step: Option<usize>,
    stale_steps: usize,
    winner: Option<usize>,
    peak_activation: f32,
}

impl RegionalDetectorRuntime {
    pub(crate) fn new(config: RegionalDetectorConfig) -> Self {
        Self {
            label: config.label,
            activation_threshold: config.activation_threshold,
            refresh_interval: config.refresh_interval,
            last_active_step: None,
            stale_steps: 0,
            winner: None,
            peak_activation: 0.0,
        }
    }

    pub(crate) fn observe(&mut self, step: usize, winner: Option<usize>, peak: f32) {
        let peak = if peak.is_finite() { peak.max(0.0) } else { 0.0 };
        self.peak_activation = peak;
        self.winner = winner;
        if peak >= self.activation_threshold && peak > 0.0 {
            self.last_active_step = Some(step);
            self.stale_steps = 0;
        } else {
            self.stale_steps = self.stale_steps.saturating_add(1);
        }
    }

    pub(crate) fn reset(&mut self) {
        self.last_active_step = None;
        self.stale_steps = 0;
        self.winner = None;
        self.peak_activation = 0.0;
    }

    pub(crate) fn state(&self) -> RegionalDetectorState {
        RegionalDetectorState {
            label: self.label.clone(),
            winner: self.winner,
            peak_activation: self.peak_activation,
            last_active_step: self.last_active_step,
            stale_steps: self.stale_steps,
            needs_refresh: self.needs_refresh(),
        }
    }

    pub(crate) fn needs_refresh(&self) -> bool {
        self.refresh_interval > 0 && self.stale_steps >= self.refresh_interval
    }
}
