use std::cmp::Ordering;

use retriever::Retriever;

use super::cosine_similarity;

#[derive(Clone, Debug)]
pub struct FactRecruitment {
    retriever: Retriever,
    facts: Vec<Vec<f32>>,
    strength: f32,
    temperature: f32,
}

impl FactRecruitment {
    pub fn new(
        retriever: Retriever,
        facts: Vec<Vec<f32>>,
        strength: f32,
        temperature: f32,
    ) -> Self {
        Self {
            retriever,
            facts,
            strength: strength.max(0.0),
            temperature: temperature.max(1e-3),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.facts.is_empty()
    }

    pub fn recruit(&self, query: &[f32]) -> Option<Vec<f32>> {
        if self.is_empty() || query.is_empty() {
            return None;
        }
        let mut blended = vec![0.0; query.len()];
        let mut weight_sum = 0.0f32;
        for (idx, similarity) in self.approximate_candidates(query) {
            let fact = self.facts.get(idx)?;
            if fact.len() != query.len() {
                continue;
            }
            let weight = ((similarity + 1.0) * 0.5)
                .clamp(0.0, 1.0)
                .powf(1.0 / self.temperature);
            if weight <= 0.0 {
                continue;
            }
            weight_sum += weight;
            for (slot, &value) in blended.iter_mut().zip(fact.iter()) {
                *slot += weight * value;
            }
        }
        if weight_sum <= 0.0 {
            return None;
        }
        for value in &mut blended {
            *value = (*value / weight_sum) * self.strength;
        }
        Some(blended)
    }

    fn approximate_candidates(&self, query: &[f32]) -> Vec<(usize, f32)> {
        let total = self.facts.len();
        if total == 0 {
            return Vec::new();
        }
        let top_k = self.retriever.top_k().max(1);
        let step = (total / (top_k * 2)).max(1);
        let mut scored: Vec<_> = (0..total)
            .step_by(step)
            .filter_map(|idx| {
                let fact = self.facts.get(idx)?;
                (fact.len() == query.len()).then(|| (idx, cosine_similarity(query, fact)))
            })
            .collect();
        if scored.is_empty() {
            return scored;
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        scored.truncate(top_k);
        if step == 1 {
            return scored;
        }
        let mut refined = scored.clone();
        let window = step.min(4);
        for &(center, _) in &scored {
            let start = center.saturating_sub(window);
            let end = (center + window).min(total - 1);
            for idx in start..=end {
                if refined.iter().any(|(seen, _)| *seen == idx) {
                    continue;
                }
                if let Some(fact) = self.facts.get(idx) {
                    if fact.len() == query.len() {
                        refined.push((idx, cosine_similarity(query, fact)));
                    }
                }
            }
        }
        refined.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        refined.truncate(top_k);
        refined
    }
}
