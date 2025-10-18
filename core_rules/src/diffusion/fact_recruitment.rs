use std::fmt;

use retriever::{MemoryRecord, Retriever, RetrieverError};

pub struct FactRecruitment {
    retriever: Retriever,
    strength: f32,
    temperature: f32,
}

impl FactRecruitment {
    pub fn new(
        mut retriever: Retriever,
        facts: Vec<Vec<f32>>,
        strength: f32,
        temperature: f32,
    ) -> Result<Self, RetrieverError> {
        if !facts.is_empty() {
            let records = facts
                .into_iter()
                .enumerate()
                .map(|(idx, fact)| MemoryRecord::new(idx as u64, fact.clone(), fact));
            retriever.ingest(records)?;
        }
        Ok(Self {
            retriever,
            strength: strength.max(0.0),
            temperature: temperature.max(1e-3),
        })
    }

    pub fn is_empty(&self) -> bool {
        self.retriever.is_empty()
    }

    pub fn recruit(&self, query: &[f32]) -> Option<Vec<f32>> {
        if self.is_empty() || query.is_empty() {
            return None;
        }
        let hits = self.retriever.search(query, None).ok()?;
        if hits.is_empty() {
            return None;
        }
        let mut blended = vec![0.0; query.len()];
        let mut weight_sum = 0.0f32;
        for hit in hits {
            let fact = self.retriever.value(hit.key)?;
            if fact.len() != query.len() {
                continue;
            }
            let weight = ((hit.similarity + 1.0) * 0.5)
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
}

impl Clone for FactRecruitment {
    fn clone(&self) -> Self {
        let snapshot = self
            .retriever
            .snapshot()
            .expect("retriever snapshot for clone");
        let retriever = Retriever::from_snapshot(self.retriever.config().clone(), snapshot)
            .expect("retriever clone");
        Self {
            retriever,
            strength: self.strength,
            temperature: self.temperature,
        }
    }
}

impl fmt::Debug for FactRecruitment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FactRecruitment")
            .field("entries", &self.retriever.len())
            .field("strength", &self.strength)
            .field("temperature", &self.temperature)
            .finish()
    }
}
