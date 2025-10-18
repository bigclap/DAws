//! Training pipeline scaffolding placeholder.

pub mod config;
pub mod dataset;
pub mod metrics;
pub mod offline;
pub mod optimizer;

pub use config::OfflineTrainerConfig;
pub use dataset::{DecoderSample, MmapDataset};
pub use metrics::{distinct_n, median_cosine_similarity};
pub use offline::{OfflineDecoderTrainer, ValidationRecord, ValidationReport};
pub use optimizer::AdamWSchedule;

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_sample(file: &mut tempfile::NamedTempFile, context: &str, target_tokens: &[&str]) {
        let record = serde_json::json!({
            "context": context,
            "target_tokens": target_tokens,
            "target_embedding": [1.0, 0.0],
            "retrieval_candidates": [[1.0, 0.0], [0.0, 1.0]]
        });
        writeln!(file, "{}", record).unwrap();
    }

    #[test]
    fn mmap_dataset_reads_json_records() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        write_sample(&mut file, "prompt", &["hello", "world"]);
        write_sample(&mut file, "query", &["foo"]);

        let dataset = MmapDataset::open(file.path()).unwrap();
        let samples: Vec<_> = dataset.iter().collect::<Result<_, _>>().unwrap();
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].context, "prompt");
        assert_eq!(samples[1].target_tokens, vec!["foo".to_string()]);
    }

    #[test]
    fn adamw_schedule_warmup_and_cosine() {
        let schedule = AdamWSchedule::new(1e-3, 1e-5, 2, 10);
        let step0 = schedule.learning_rate(0);
        let step2 = schedule.learning_rate(2);
        assert!(step0 > 0.0 && step0 < schedule.base_lr);
        assert!(step2 < schedule.base_lr);
    }

    #[test]
    fn offline_validation_reports_metrics() {
        let trainer = OfflineDecoderTrainer::new(OfflineTrainerConfig {
            validation_top_ks: vec![1, 2],
            ..OfflineTrainerConfig::default()
        });

        let record_one = ValidationRecord {
            predicted_embedding: vec![1.0, 0.0],
            target_embedding: vec![1.0, 0.0],
            retrieval_candidates: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            generated_tokens: vec!["a".into(), "b".into()],
            target_tokens: vec!["a".into(), "b".into()],
            log_probs: vec![-0.1, -0.1],
        };

        let record_two = ValidationRecord {
            predicted_embedding: vec![0.0, 1.0],
            target_embedding: vec![1.0, 0.0],
            retrieval_candidates: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            generated_tokens: vec!["a".into(), "c".into()],
            target_tokens: vec!["a".into(), "d".into()],
            log_probs: vec![-0.5, -1.0],
        };

        let report = trainer
            .validate(vec![record_one, record_two])
            .expect("validation report");

        assert!(report.cosine_at_median >= -1.0 && report.cosine_at_median <= 1.0);
        assert!(report.retrieval_rank_at_k.get(&1).is_some());
        assert!(report.distinct_n > 0.0);
        assert!(report.ppl_surrogate > 0.0);
    }
}
