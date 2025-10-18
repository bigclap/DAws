use std::cmp::Ordering;
use std::collections::HashMap;

use hnsw_rs::prelude::{DistCosine, Hnsw};
use thiserror::Error;

use crate::config::{MemoryHit, MemoryRecord, RetrieverConfig};
use crate::io::{KvMemorySnapshot, MemorySnapshotError};

/// Errors that can occur while operating the retriever.
#[derive(Debug, Error)]
pub enum RetrieverError {
    /// Invalid configuration parameter was supplied.
    #[error("invalid configuration: {0}")]
    InvalidConfiguration(&'static str),
    /// Provided vector does not match the configured dimension.
    #[error("expected embedding dimension {expected}, found {found}")]
    DimensionMismatch { expected: usize, found: usize },
    /// Provided value vector does not match the configured dimension.
    #[error("expected value dimension {expected}, found {found}")]
    ValueDimensionMismatch { expected: usize, found: usize },
    /// Attempted to ingest more entries than allowed by the configuration.
    #[error("capacity exceeded: {capacity} < {requested}")]
    CapacityExceeded { capacity: usize, requested: usize },
    /// Attempted to insert a key that already exists in the memory.
    #[error("key {key} already exists in the retriever")]
    DuplicateKey { key: u64 },
    /// Encountered a zero-norm embedding vector which cannot be normalized.
    #[error("embedding for key {key} has zero norm")]
    ZeroVector { key: u64 },
    /// Query vector had zero norm after validation.
    #[error("query embedding has zero norm")]
    ZeroQuery,
    /// Snapshot loading or writing failed.
    #[error(transparent)]
    Snapshot(#[from] MemorySnapshotError),
}

/// High-level wrapper around a cosine-distance HNSW index.
pub struct Retriever {
    config: RetrieverConfig,
    index: Hnsw<'static, f32, DistCosine>,
    records: Vec<MemoryRecord>,
    offsets: HashMap<u64, usize>,
    gates: Vec<f32>,
}

impl Retriever {
    /// Constructs an empty retriever using the supplied configuration.
    pub fn new(config: RetrieverConfig) -> Result<Self, RetrieverError> {
        validate_config(&config)?;
        let index = Hnsw::new(
            config.max_connections,
            config.max_elements,
            config.max_layers,
            config.ef_construction,
            DistCosine {},
        );
        Ok(Self {
            config,
            index,
            records: Vec::new(),
            offsets: HashMap::new(),
            gates: Vec::new(),
        })
    }

    /// Returns the configured default number of neighbours per query.
    pub fn top_k(&self) -> usize {
        self.config.top_k.max(1)
    }

    /// Returns the underlying configuration.
    pub fn config(&self) -> &RetrieverConfig {
        &self.config
    }

    /// Returns the number of stored memory records.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Reports whether the retriever currently contains no memories.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Inserts new memory records into the ANN index.
    pub fn ingest<I>(&mut self, records: I) -> Result<(), RetrieverError>
    where
        I: IntoIterator<Item = MemoryRecord>,
    {
        let staged: Vec<MemoryRecord> = records.into_iter().collect();
        if staged.is_empty() {
            return Ok(());
        }
        let current = self.records.len();
        let requested = current + staged.len();
        if requested > self.config.max_elements {
            return Err(RetrieverError::CapacityExceeded {
                capacity: self.config.max_elements,
                requested,
            });
        }
        for record in &staged {
            if record.embedding.len() != self.config.dimension {
                return Err(RetrieverError::DimensionMismatch {
                    expected: self.config.dimension,
                    found: record.embedding.len(),
                });
            }
            if record.value.len() != self.config.value_dimension {
                return Err(RetrieverError::ValueDimensionMismatch {
                    expected: self.config.value_dimension,
                    found: record.value.len(),
                });
            }
            if self.offsets.contains_key(&record.key) {
                return Err(RetrieverError::DuplicateKey { key: record.key });
            }
        }

        for mut record in staged {
            let normalized = normalize(&record.embedding, Some(record.key))?;
            record.embedding = normalized;
            let data_id = self.records.len();
            self.index.insert((&record.embedding, data_id));
            self.offsets.insert(record.key, data_id);
            self.records.push(record);
            self.gates.push(self.initial_gate());
        }
        Ok(())
    }

    /// Executes a cosine-similarity ANN search over the stored embeddings.
    pub fn search(
        &mut self,
        query: &[f32],
        limit: Option<usize>,
    ) -> Result<Vec<MemoryHit>, RetrieverError> {
        if query.len() != self.config.dimension {
            return Err(RetrieverError::DimensionMismatch {
                expected: self.config.dimension,
                found: query.len(),
            });
        }
        if self.records.is_empty() {
            self.decay_gates();
            return Ok(Vec::new());
        }
        self.decay_gates();
        let normalized_query = normalize(query, None)?;
        let limit = limit
            .unwrap_or_else(|| self.top_k())
            .clamp(1, self.records.len());
        let ef = self.config.ef_search.max(limit);
        let mut results = self
            .index
            .search(&normalized_query, limit, ef)
            .into_iter()
            .filter_map(|neighbor| {
                let record = self.records.get(neighbor.d_id)?;
                let gate = self
                    .gates
                    .get(neighbor.d_id)
                    .copied()
                    .unwrap_or(self.config.gate_floor);
                let similarity =
                    cosine(&normalized_query, &record.embedding) * gate.clamp(0.0, f32::MAX);
                Some(MemoryHit {
                    key: record.key,
                    similarity,
                    gate,
                })
            })
            .collect::<Vec<_>>();
        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(Ordering::Equal)
        });
        results.truncate(limit);
        self.refresh_hits(&results);
        Ok(results)
    }

    /// Retrieves the stored value for a given memory key.
    pub fn value(&self, key: u64) -> Option<&[f32]> {
        self.offsets
            .get(&key)
            .and_then(|&idx| self.records.get(idx))
            .map(|record| record.value.as_slice())
    }

    /// Returns the current gate value for the supplied key, if present.
    pub fn gate(&self, key: u64) -> Option<f32> {
        self.offsets
            .get(&key)
            .and_then(|&idx| self.gates.get(idx))
            .copied()
    }

    /// Creates a serialisable snapshot of the retriever state.
    pub fn snapshot(&self) -> Result<KvMemorySnapshot, RetrieverError> {
        let mut vectors = Vec::with_capacity(self.records.len() * self.config.dimension);
        let mut values = Vec::with_capacity(self.records.len() * self.config.value_dimension);
        let mut keys = Vec::with_capacity(self.records.len());
        for record in &self.records {
            keys.push(record.key);
            vectors.extend_from_slice(&record.embedding);
            values.extend_from_slice(&record.value);
        }
        Ok(KvMemorySnapshot::new(
            Vec::new(),
            keys,
            vectors,
            values,
            self.records.len(),
            self.config.dimension,
            self.config.value_dimension,
        )?)
    }

    /// Reconstructs a retriever from a snapshot.
    pub fn from_snapshot(
        mut config: RetrieverConfig,
        snapshot: KvMemorySnapshot,
    ) -> Result<Self, RetrieverError> {
        if config.max_elements < snapshot.num_vectors {
            config.max_elements = snapshot.num_vectors.max(1);
        }
        let mut retriever = Self::new(config)?;
        let mut embeddings = snapshot.vectors.chunks(snapshot.dimension);
        let mut values = snapshot.values.chunks(snapshot.value_dimension);
        let records = snapshot
            .keys
            .into_iter()
            .map(|key| MemoryRecord {
                key,
                embedding: embeddings.next().expect("embedding chunk present").to_vec(),
                value: values.next().expect("value chunk present").to_vec(),
            })
            .collect::<Vec<_>>();
        retriever.ingest(records)?;
        Ok(retriever)
    }

    /// Applies a refresh pulse to the supplied memory keys.
    pub fn refresh_pulse(&mut self, keys: &[u64]) {
        if self.records.is_empty() {
            return;
        }
        self.decay_gates();
        let refresh = self.refresh_gate_value();
        for key in keys {
            if let Some(&idx) = self.offsets.get(key) {
                if let Some(gate) = self.gates.get_mut(idx) {
                    *gate = refresh;
                }
            }
        }
    }

    fn refresh_hits(&mut self, hits: &[MemoryHit]) {
        if hits.is_empty() {
            return;
        }
        let refresh = self.refresh_gate_value();
        for hit in hits {
            if let Some(&idx) = self.offsets.get(&hit.key) {
                if let Some(gate) = self.gates.get_mut(idx) {
                    *gate = refresh;
                }
            }
        }
    }

    fn decay_gates(&mut self) {
        if self.gates.is_empty() {
            return;
        }
        let decay = self.config.gate_decay.clamp(0.0, 1.0);
        let floor = self.config.gate_floor;
        for gate in &mut self.gates {
            *gate = (*gate * decay).max(floor);
            if *gate > self.config.gate_ceiling {
                *gate = self.config.gate_ceiling;
            }
        }
    }

    fn refresh_gate_value(&self) -> f32 {
        self.config
            .gate_refresh
            .clamp(self.config.gate_floor, self.config.gate_ceiling)
    }

    fn initial_gate(&self) -> f32 {
        self.refresh_gate_value()
    }
}

fn validate_config(config: &RetrieverConfig) -> Result<(), RetrieverError> {
    if config.dimension == 0 {
        return Err(RetrieverError::InvalidConfiguration(
            "embedding dimension must be greater than zero",
        ));
    }
    if config.value_dimension == 0 {
        return Err(RetrieverError::InvalidConfiguration(
            "value dimension must be greater than zero",
        ));
    }
    if config.max_elements == 0 {
        return Err(RetrieverError::InvalidConfiguration(
            "max_elements must be greater than zero",
        ));
    }
    if config.max_layers == 0 {
        return Err(RetrieverError::InvalidConfiguration(
            "max_layers must be greater than zero",
        ));
    }
    if config.ef_search == 0 {
        return Err(RetrieverError::InvalidConfiguration(
            "ef_search must be greater than zero",
        ));
    }
    if !(0.0..=1.0).contains(&config.gate_decay) {
        return Err(RetrieverError::InvalidConfiguration(
            "gate_decay must be within [0, 1]",
        ));
    }
    if config.gate_ceiling < config.gate_floor {
        return Err(RetrieverError::InvalidConfiguration(
            "gate_ceiling must be greater than or equal to gate_floor",
        ));
    }
    if config.gate_refresh < config.gate_floor || config.gate_refresh > config.gate_ceiling {
        return Err(RetrieverError::InvalidConfiguration(
            "gate_refresh must lie within [gate_floor, gate_ceiling]",
        ));
    }
    Ok(())
}

fn normalize(vector: &[f32], key: Option<u64>) -> Result<Vec<f32>, RetrieverError> {
    let norm = vector
        .iter()
        .map(|v| (*v as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    if norm <= f64::EPSILON {
        return match key {
            Some(key) => Err(RetrieverError::ZeroVector { key }),
            None => Err(RetrieverError::ZeroQuery),
        };
    }
    Ok(vector.iter().map(|v| (*v as f64 / norm) as f32).collect())
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum::<f64>() as f32
}

#[cfg(test)]
mod tests;
