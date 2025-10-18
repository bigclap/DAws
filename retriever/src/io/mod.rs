//! Persistence helpers for approximate nearest neighbour memory stores.

mod files;
mod metadata;
mod tensors;

use std::fs;
use std::path::Path;

use metadata::{read_metadata, write_metadata};
use tensors::{
    read_f32_matrix, read_f32_vector, read_keys, write_f32_matrix, write_f32_vector, write_keys,
};

use thiserror::Error;

use files::{read_index, write_index};

pub use metadata::MemoryMetadata;

const VECTORS_TENSOR: &str = "vectors";
const VALUES_TENSOR: &str = "values";
const KEYS_TENSOR: &str = "keys";
const GATES_TENSOR: &str = "gates";

const INDEX_FILE: &str = "hnsw.index";
const VECTORS_FILE: &str = "vecs.st";
const VALUES_FILE: &str = "vals.st";
const KEYS_FILE: &str = "keys.st";
const META_FILE: &str = "meta.parquet";
const GATES_FILE: &str = "gates.st";

/// Snapshot of the persistent ANN memory files.
#[derive(Clone, Debug, PartialEq)]
pub struct KvMemorySnapshot {
    /// Raw serialized HNSW index bytes.
    pub index: Vec<u8>,
    /// Keys associated with each stored embedding.
    pub keys: Vec<u64>,
    /// Flattened embedding payload stored row-major.
    pub vectors: Vec<f32>,
    /// Flattened value payload stored row-major.
    pub values: Vec<f32>,
    /// Per-record memory gate values.
    pub gates: Vec<f32>,
    /// Number of vectors contained in the payload.
    pub num_vectors: usize,
    /// Dimensionality of each embedding vector.
    pub dimension: usize,
    /// Dimensionality of the value vectors.
    pub value_dimension: usize,
}

impl KvMemorySnapshot {
    /// Constructs a snapshot validating the provided payload sizes.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        index: Vec<u8>,
        keys: Vec<u64>,
        vectors: Vec<f32>,
        values: Vec<f32>,
        gates: Vec<f32>,
        num_vectors: usize,
        dimension: usize,
        value_dimension: usize,
    ) -> Result<Self, MemorySnapshotError> {
        let expected_vectors =
            num_vectors
                .checked_mul(dimension)
                .ok_or(MemorySnapshotError::InvalidShape {
                    expected: num_vectors,
                    found: dimension,
                })?;
        if vectors.len() != expected_vectors {
            return Err(MemorySnapshotError::InvalidShape {
                expected: expected_vectors,
                found: vectors.len(),
            });
        }
        let expected_values =
            num_vectors
                .checked_mul(value_dimension)
                .ok_or(MemorySnapshotError::InvalidShape {
                    expected: num_vectors,
                    found: value_dimension,
                })?;
        if values.len() != expected_values {
            return Err(MemorySnapshotError::InvalidShape {
                expected: expected_values,
                found: values.len(),
            });
        }
        if keys.len() != num_vectors {
            return Err(MemorySnapshotError::InvalidShape {
                expected: num_vectors,
                found: keys.len(),
            });
        }
        if !gates.is_empty() && gates.len() != num_vectors {
            return Err(MemorySnapshotError::InvalidShape {
                expected: num_vectors,
                found: gates.len(),
            });
        }
        Ok(Self {
            index,
            keys,
            vectors,
            values,
            gates,
            num_vectors,
            dimension,
            value_dimension,
        })
    }

    /// Persists the snapshot to the supplied directory.
    pub fn write_all<P: AsRef<Path>>(&self, dir: P) -> Result<(), MemorySnapshotError> {
        let dir = dir.as_ref();
        fs::create_dir_all(dir)?;
        write_index(dir.join(INDEX_FILE), &self.index)?;
        write_f32_matrix(
            dir.join(VECTORS_FILE),
            VECTORS_TENSOR,
            &self.vectors,
            self.num_vectors,
            self.dimension,
        )?;
        write_f32_matrix(
            dir.join(VALUES_FILE),
            VALUES_TENSOR,
            &self.values,
            self.num_vectors,
            self.value_dimension,
        )?;
        write_keys(dir.join(KEYS_FILE), KEYS_TENSOR, &self.keys)?;
        if !self.gates.is_empty() {
            write_f32_vector(dir.join(GATES_FILE), GATES_TENSOR, &self.gates)?;
        }
        write_metadata(
            dir.join(META_FILE),
            &MemoryMetadata {
                num_vectors: self.num_vectors,
                dimension: self.dimension,
                value_dimension: self.value_dimension,
            },
        )?;
        Ok(())
    }

    /// Loads a snapshot from the supplied directory.
    pub fn read_all<P: AsRef<Path>>(dir: P) -> Result<Self, MemorySnapshotError> {
        let dir = dir.as_ref();
        let index = read_index(dir.join(INDEX_FILE))?;
        let (vectors, vec_rows, vec_dims) =
            read_f32_matrix(dir.join(VECTORS_FILE), VECTORS_TENSOR)?;
        let (values, val_rows, val_dims) = read_f32_matrix(dir.join(VALUES_FILE), VALUES_TENSOR)?;
        let keys = read_keys(dir.join(KEYS_FILE), KEYS_TENSOR)?;
        let meta = read_metadata(dir.join(META_FILE))?;
        let gates = match read_f32_vector(dir.join(GATES_FILE), GATES_TENSOR) {
            Ok((data, len)) => {
                if len != meta.num_vectors {
                    return Err(MemorySnapshotError::InvalidShape {
                        expected: meta.num_vectors,
                        found: len,
                    });
                }
                data
            }
            Err(MemorySnapshotError::Io(err)) if err.kind() == std::io::ErrorKind::NotFound => {
                Vec::new()
            }
            Err(err) => return Err(err),
        };
        if vec_rows != meta.num_vectors
            || vec_dims != meta.dimension
            || val_rows != meta.num_vectors
            || val_dims != meta.value_dimension
        {
            return Err(MemorySnapshotError::InconsistentPayload);
        }
        Self::new(
            index,
            keys,
            vectors,
            values,
            gates,
            meta.num_vectors,
            meta.dimension,
            meta.value_dimension,
        )
    }
}

/// Errors that may arise while interacting with memory snapshots.
#[derive(Debug, Error)]
pub enum MemorySnapshotError {
    /// Filesystem interaction failed.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// safetensors encoding or decoding failed.
    #[error("safetensors error: {0}")]
    SafeTensors(#[from] safetensors::SafeTensorError),
    /// Parquet writer or reader reported an error.
    #[error("parquet error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),
    /// Metadata file did not contain any rows.
    #[error("metadata file is empty")]
    EmptyMetadata,
    /// Metadata contained negative or otherwise invalid values.
    #[error("invalid metadata value {0}")]
    InvalidMetadata(i64),
    /// Metadata value exceeded the supported range.
    #[error("metadata value {value} exceeds parquet limits")]
    MetadataOverflow { value: usize },
    /// Encountered an unsupported tensor dtype in the safetensors payload.
    #[error("unsupported tensor dtype {dtype:?}")]
    UnsupportedDType { dtype: safetensors::Dtype },
    /// Vector payload length does not match the provided shape.
    #[error("vector payload length mismatch (expected {expected}, found {found})")]
    InvalidShape { expected: usize, found: usize },
    /// Persisted components disagree on the recorded dimensions.
    #[error("vector payload inconsistent with metadata")]
    InconsistentPayload,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn snapshot_round_trip() {
        let snapshot = KvMemorySnapshot::new(
            vec![1, 2, 3],
            vec![7, 8],
            vec![0.1, 0.2, 0.3, 0.4],
            vec![0.5, 0.6, 0.7, 0.8],
            vec![0.9, 0.4],
            2,
            2,
            2,
        )
        .unwrap();
        let dir = tempdir().unwrap();
        snapshot.write_all(dir.path()).unwrap();
        let restored = KvMemorySnapshot::read_all(dir.path()).unwrap();
        assert_eq!(restored, snapshot);
    }

    #[test]
    fn metadata_inconsistency_is_detected() {
        let dir = tempdir().unwrap();
        let path = dir.path();
        write_index(path.join(INDEX_FILE), &[0]).unwrap();
        write_f32_matrix(
            path.join(VECTORS_FILE),
            VECTORS_TENSOR,
            &[1.0, 2.0, 3.0, 4.0],
            2,
            2,
        )
        .unwrap();
        write_f32_matrix(
            path.join(VALUES_FILE),
            VALUES_TENSOR,
            &[1.0, 2.0, 3.0, 4.0],
            2,
            2,
        )
        .unwrap();
        write_keys(path.join(KEYS_FILE), KEYS_TENSOR, &[1, 2]).unwrap();
        write_metadata(
            path.join(META_FILE),
            &MemoryMetadata {
                num_vectors: 3,
                dimension: 2,
                value_dimension: 2,
            },
        )
        .unwrap();
        let err = KvMemorySnapshot::read_all(path).unwrap_err();
        assert!(matches!(err, MemorySnapshotError::InconsistentPayload));
    }

    #[test]
    fn gates_file_is_optional_for_backwards_compatibility() {
        let dir = tempdir().unwrap();
        let path = dir.path();
        write_index(path.join(INDEX_FILE), &[0]).unwrap();
        write_f32_matrix(path.join(VECTORS_FILE), VECTORS_TENSOR, &[1.0, 0.0], 1, 2).unwrap();
        write_f32_matrix(path.join(VALUES_FILE), VALUES_TENSOR, &[1.0, 0.0], 1, 2).unwrap();
        write_keys(path.join(KEYS_FILE), KEYS_TENSOR, &[1]).unwrap();
        write_metadata(
            path.join(META_FILE),
            &MemoryMetadata {
                num_vectors: 1,
                dimension: 2,
                value_dimension: 2,
            },
        )
        .unwrap();
        let snapshot = KvMemorySnapshot::read_all(path).unwrap();
        assert!(snapshot.gates.is_empty());
    }
}
