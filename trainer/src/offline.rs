use crate::OfflineTrainerConfig;
use crate::metrics::{cosine_similarity, distinct_n, median_cosine_similarity};
use anyhow::{Result, anyhow, ensure};
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct ValidationRecord {
    pub predicted_embedding: Vec<f32>,
    pub target_embedding: Vec<f32>,
    pub retrieval_candidates: Vec<Vec<f32>>,
    pub generated_tokens: Vec<String>,
    pub target_tokens: Vec<String>,
    pub log_probs: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValidationReport {
    pub cosine_at_median: f32,
    pub retrieval_rank_at_k: BTreeMap<usize, f32>,
    pub distinct_n: f32,
    pub ppl_surrogate: f32,
}

pub struct OfflineDecoderTrainer {
    config: OfflineTrainerConfig,
}

impl OfflineDecoderTrainer {
    pub fn new(config: OfflineTrainerConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &OfflineTrainerConfig {
        &self.config
    }

    pub fn validate<I>(&self, records: I) -> Result<ValidationReport>
    where
        I: IntoIterator<Item = ValidationRecord>,
    {
        let mut cosines = Vec::new();
        let mut distinct_scores = Vec::new();
        let mut ppl_scores = Vec::new();
        let mut success_counts: BTreeMap<usize, usize> = self
            .config
            .validation_top_ks
            .iter()
            .copied()
            .map(|k| (k, 0))
            .collect();

        let mut total_records = 0usize;

        for record in records.into_iter() {
            ensure!(
                record.predicted_embedding.len() == record.target_embedding.len(),
                "embedding length mismatch"
            );
            ensure!(
                !record.retrieval_candidates.is_empty(),
                "retrieval candidates must not be empty"
            );
            ensure!(
                !record.log_probs.is_empty(),
                "log probs required for ppl surrogate"
            );

            total_records += 1;

            let cosine = cosine_similarity(&record.predicted_embedding, &record.target_embedding);
            cosines.push(cosine);

            let candidate_scores: Vec<f32> = record
                .retrieval_candidates
                .iter()
                .map(|candidate| cosine_similarity(&record.predicted_embedding, candidate))
                .collect();

            let target_score = candidate_scores[0];
            let mut rank = 1usize;
            for score in candidate_scores.iter().skip(1) {
                if score > &target_score {
                    rank += 1;
                }
            }

            for (k, count) in success_counts.iter_mut() {
                if rank <= *k {
                    *count += 1;
                }
            }

            distinct_scores.push(distinct_n(&record.generated_tokens, 3));
            let mean_log_prob =
                record.log_probs.iter().sum::<f32>() / record.log_probs.len() as f32;
            let ppl = (-mean_log_prob).exp();
            ppl_scores.push(ppl);
        }

        if total_records == 0 {
            return Err(anyhow!("no validation records supplied"));
        }

        let cosine_at_median = median_cosine_similarity(&cosines);
        let retrieval_rank_at_k = success_counts
            .into_iter()
            .map(|(k, count)| (k, count as f32 / total_records as f32))
            .collect();
        let distinct_n = distinct_scores.iter().sum::<f32>() / distinct_scores.len() as f32;
        let ppl_surrogate = ppl_scores.iter().sum::<f32>() / ppl_scores.len() as f32;

        Ok(ValidationReport {
            cosine_at_median,
            retrieval_rank_at_k,
            distinct_n,
            ppl_surrogate,
        })
    }
}
