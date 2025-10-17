//! Training pipeline scaffolding placeholder.

/// Minimal training configuration used to prove wiring between crates.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TrainerConfig {
    /// Learning rate placeholder to match future design docs.
    pub learning_rate: f32,
}

/// No-op trainer that tracks configuration values.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Trainer {
    config: TrainerConfig,
}

impl Trainer {
    /// Creates a new trainer with the supplied [`TrainerConfig`].
    pub fn new(config: TrainerConfig) -> Self {
        Self { config }
    }

    /// Returns the configured learning rate.
    pub fn learning_rate(&self) -> f32 {
        self.config.learning_rate
    }
}

#[cfg(test)]
mod tests {
    use super::{Trainer, TrainerConfig};

    #[test]
    fn learning_rate_round_trip() {
        let trainer = Trainer::new(TrainerConfig { learning_rate: 0.1 });
        assert_eq!(trainer.learning_rate(), 0.1);
    }
}
