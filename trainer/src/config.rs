use serde::{Deserialize, Serialize};

/// Configuration for the offline decoder trainer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OfflineTrainerConfig {
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub warmup_steps: usize,
    pub total_steps: usize,
    pub min_lr: f32,
    pub mixed_precision: bool,
    pub validation_top_ks: Vec<usize>,
}

impl Default for OfflineTrainerConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            weight_decay: 0.01,
            warmup_steps: 100,
            total_steps: 10_000,
            min_lr: 1e-5,
            mixed_precision: false,
            validation_top_ks: vec![1, 5, 10],
        }
    }
}
