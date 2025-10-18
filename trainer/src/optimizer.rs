pub struct AdamWSchedule {
    pub base_lr: f32,
    pub min_lr: f32,
    pub warmup_steps: usize,
    pub total_steps: usize,
}

impl AdamWSchedule {
    pub fn new(base_lr: f32, min_lr: f32, warmup_steps: usize, total_steps: usize) -> Self {
        Self {
            base_lr,
            min_lr,
            warmup_steps,
            total_steps,
        }
    }

    pub fn learning_rate(&self, step: usize) -> f32 {
        if self.total_steps == 0 {
            return self.min_lr;
        }

        if self.warmup_steps > 0 && step < self.warmup_steps {
            let warmup_progress = (step + 1) as f32 / self.warmup_steps as f32;
            return self.base_lr * warmup_progress;
        }

        let effective_total = self.total_steps.saturating_sub(self.warmup_steps).max(1);
        let progress =
            (step.saturating_sub(self.warmup_steps) as f32 + 1.0) / (effective_total as f32 + 1.0);
        let clamped = progress.min(1.0);
        let cosine = (std::f32::consts::PI * clamped).cos();
        let lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + cosine);
        lr.max(self.min_lr)
    }
}
