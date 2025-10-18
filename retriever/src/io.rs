//! Persistence helpers for approximate nearest neighbour memory stores.

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use bytemuck::cast_slice;
use parquet::data_type::Int64Type;
use parquet::file::properties::WriterProperties;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::file::writer::SerializedFileWriter;
use parquet::record::RowAccessor;
use parquet::schema::parser::parse_message_type;
use safetensors::tensor::TensorView;
use safetensors::{SafeTensorError, SafeTensors, serialize};
use thiserror::Error;

const VECTORS_TENSOR: &str = "vectors";
const INDEX_FILE: &str = "hnsw.index";
const VECTORS_FILE: &str = "vecs.st";
const META_FILE: &str = "meta.parquet";

/// Metadata stored alongside the ANN memory vectors.
#[derive(Clone, Debug, PartialEq)]
pub struct MemoryMetadata {
    /// Number of vector entries persisted in the shard.
    pub num_vectors: usize,
    /// Dimensionality of each vector.
    pub dimension: usize,
}

/// Snapshot of the persistent ANN memory files.
#[derive(Clone, Debug, PartialEq)]
pub struct KvMemorySnapshot {
    /// Raw serialized HNSW index bytes.
    pub index: Vec<u8>,
    /// Flattened vector payload stored row-major.
    pub vectors: Vec<f32>,
    /// Number of vectors contained in the payload.
    pub num_vectors: usize,
    /// Dimensionality of each vector.
    pub dimension: usize,
}

impl KvMemorySnapshot {
    /// Constructs a snapshot validating the provided payload size.
    pub fn new(
        index: Vec<u8>,
        vectors: Vec<f32>,
        num_vectors: usize,
        dimension: usize,
    ) -> Result<Self, MemorySnapshotError> {
        let expected =
            num_vectors
                .checked_mul(dimension)
                .ok_or(MemorySnapshotError::InvalidShape {
                    expected: num_vectors,
                    found: dimension,
                })?;
        if vectors.len() != expected {
            return Err(MemorySnapshotError::InvalidShape {
                expected,
                found: vectors.len(),
            });
        }
        Ok(Self {
            index,
            vectors,
            num_vectors,
            dimension,
        })
    }

    /// Persists the snapshot to the supplied directory.
    pub fn write_all<P: AsRef<Path>>(&self, dir: P) -> Result<(), MemorySnapshotError> {
        let dir = dir.as_ref();
        fs::create_dir_all(dir)?;
        write_index(dir.join(INDEX_FILE), &self.index)?;
        write_vectors(
            dir.join(VECTORS_FILE),
            &self.vectors,
            self.num_vectors,
            self.dimension,
        )?;
        write_metadata(
            dir.join(META_FILE),
            &MemoryMetadata {
                num_vectors: self.num_vectors,
                dimension: self.dimension,
            },
        )?;
        Ok(())
    }

    /// Loads a snapshot from the supplied directory.
    pub fn read_all<P: AsRef<Path>>(dir: P) -> Result<Self, MemorySnapshotError> {
        let dir = dir.as_ref();
        let index = read_index(dir.join(INDEX_FILE))?;
        let (vectors, rows, dims) = read_vectors(dir.join(VECTORS_FILE))?;
        let meta = read_metadata(dir.join(META_FILE))?;
        if rows != meta.num_vectors || dims != meta.dimension {
            return Err(MemorySnapshotError::InconsistentPayload);
        }
        Self::new(index, vectors, meta.num_vectors, meta.dimension)
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
    SafeTensors(#[from] SafeTensorError),
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

#[inline]
fn usize_to_i64(value: usize) -> Result<i64, MemorySnapshotError> {
    if value > i64::MAX as usize {
        Err(MemorySnapshotError::MetadataOverflow { value })
    } else {
        Ok(value as i64)
    }
}

fn write_index(path: PathBuf, bytes: &[u8]) -> Result<(), MemorySnapshotError> {
    let mut file = File::create(path)?;
    file.write_all(bytes)?;
    Ok(())
}

fn read_index(path: PathBuf) -> Result<Vec<u8>, MemorySnapshotError> {
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;
    Ok(bytes)
}

fn write_vectors(
    path: PathBuf,
    vectors: &[f32],
    rows: usize,
    dims: usize,
) -> Result<(), MemorySnapshotError> {
    let expected = rows
        .checked_mul(dims)
        .ok_or_else(|| MemorySnapshotError::InvalidShape {
            expected: rows,
            found: dims,
        })?;
    if expected != vectors.len() {
        return Err(MemorySnapshotError::InvalidShape {
            expected,
            found: vectors.len(),
        });
    }
    let tensor = TensorView::new(
        safetensors::Dtype::F32,
        vec![rows, dims],
        cast_slice(vectors),
    )?;
    let bytes = serialize(std::iter::once((VECTORS_TENSOR.to_string(), tensor)), &None)?;
    let mut file = File::create(path)?;
    file.write_all(&bytes)?;
    Ok(())
}

fn read_vectors(path: PathBuf) -> Result<(Vec<f32>, usize, usize), MemorySnapshotError> {
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let tensor = tensors.tensor(VECTORS_TENSOR)?;
    if tensor.dtype() != safetensors::Dtype::F32 {
        return Err(MemorySnapshotError::UnsupportedDType {
            dtype: tensor.dtype(),
        });
    }
    let shape = tensor.shape();
    if shape.len() != 2 {
        return Err(MemorySnapshotError::InvalidShape {
            expected: 2,
            found: shape.len(),
        });
    }
    let rows = shape[0];
    let dims = shape[1];
    let data = cast_slice::<u8, f32>(tensor.data()).to_vec();
    Ok((data, rows, dims))
}

fn write_metadata(path: PathBuf, metadata: &MemoryMetadata) -> Result<(), MemorySnapshotError> {
    let schema = Arc::new(parse_message_type(
        "message kv_meta {\n  REQUIRED INT64 num_vectors;\n  REQUIRED INT64 dimension;\n}",
    )?);
    let props = Arc::new(WriterProperties::builder().build());
    let file = File::create(path)?;
    let mut writer = SerializedFileWriter::new(file, schema, props)?;
    {
        let mut row_group = writer.next_row_group()?;
        if let Some(mut col) = row_group.next_column()? {
            let writer = col.typed::<Int64Type>();
            let values = [usize_to_i64(metadata.num_vectors)?];
            writer.write_batch(&values, None, None)?;
            col.close()?;
        }
        if let Some(mut col) = row_group.next_column()? {
            let writer = col.typed::<Int64Type>();
            let values = [usize_to_i64(metadata.dimension)?];
            writer.write_batch(&values, None, None)?;
            col.close()?;
        }
        row_group.close()?;
    }
    writer.close()?;
    Ok(())
}

fn read_metadata(path: PathBuf) -> Result<MemoryMetadata, MemorySnapshotError> {
    let file = File::open(path)?;
    let reader = SerializedFileReader::new(file)?;
    let mut rows = reader.get_row_iter(None)?;
    let row = match rows.next() {
        Some(result) => result?,
        None => return Err(MemorySnapshotError::EmptyMetadata),
    };
    let num_vectors = row.get_long(0)?;
    let dimension = row.get_long(1)?;
    if num_vectors < 0 {
        return Err(MemorySnapshotError::InvalidMetadata(num_vectors));
    }
    if dimension < 0 {
        return Err(MemorySnapshotError::InvalidMetadata(dimension));
    }
    Ok(MemoryMetadata {
        num_vectors: usize::try_from(num_vectors)
            .map_err(|_| MemorySnapshotError::InvalidMetadata(num_vectors))?,
        dimension: usize::try_from(dimension)
            .map_err(|_| MemorySnapshotError::InvalidMetadata(dimension))?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn snapshot_round_trip() {
        let snapshot =
            KvMemorySnapshot::new(vec![1, 2, 3], vec![0.1, 0.2, 0.3, 0.4], 2, 2).unwrap();
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
        write_vectors(path.join(VECTORS_FILE), &[1.0f32, 2.0, 3.0, 4.0], 2, 2).unwrap();
        write_metadata(
            path.join(META_FILE),
            &MemoryMetadata {
                num_vectors: 3,
                dimension: 2,
            },
        )
        .unwrap();
        let err = KvMemorySnapshot::read_all(path).unwrap_err();
        assert!(matches!(err, MemorySnapshotError::InconsistentPayload));
    }
}
