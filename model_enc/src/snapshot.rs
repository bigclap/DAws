//! Utilities for persisting encoder embedding tables to safetensors snapshots.

use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use bytemuck::cast_slice;
use candle_core::{Device, Tensor};
use safetensors::tensor::TensorView;
use safetensors::{SafeTensorError, SafeTensors, serialize};
use thiserror::Error;

/// Errors that can occur while reading or writing embedding snapshots.
#[derive(Debug, Error)]
pub enum EmbeddingSnapshotError {
    /// Wrapper around IO failures when accessing the snapshot file.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// Raised when the safetensors payload is malformed.
    #[error("safetensors error: {0}")]
    SafeTensors(#[from] SafeTensorError),
    /// Raised when tensor conversion fails.
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
    /// Raised when the provided tensor is not two dimensional.
    #[error("embedding tensor must be two dimensional")]
    InvalidShape,
    /// Raised when the safetensors tensor has an unexpected dtype.
    #[error("expected f32 tensor but found {0:?}")]
    InvalidDType(safetensors::Dtype),
}

/// Snapshot storing a single embedding table backed by f32 weights.
#[derive(Clone, Debug, PartialEq)]
pub struct EmbeddingSnapshot {
    rows: usize,
    dims: usize,
    data: Vec<f32>,
}

impl EmbeddingSnapshot {
    const TENSOR_NAME: &'static str = "embeddings";

    /// Creates a snapshot from a Candle tensor.
    pub fn from_tensor(tensor: &Tensor) -> Result<Self, EmbeddingSnapshotError> {
        if tensor.dims().len() != 2 {
            return Err(EmbeddingSnapshotError::InvalidShape);
        }
        let rows = tensor.dims()[0];
        let dims = tensor.dims()[1];
        let data = tensor.to_vec2::<f32>()?.into_iter().flatten().collect();
        Ok(Self { rows, dims, data })
    }

    /// Restores the snapshot into a tensor located on the supplied device.
    pub fn to_tensor(&self, device: &Device) -> Result<Tensor, candle_core::Error> {
        Tensor::from_vec(self.data.clone(), (self.rows, self.dims), device)
    }

    /// Writes the snapshot to a `.stt` safetensors file.
    pub fn write<P: AsRef<Path>>(&self, path: P) -> Result<(), EmbeddingSnapshotError> {
        let path = path.as_ref();
        let tensor_view = TensorView::new(
            safetensors::Dtype::F32,
            vec![self.rows, self.dims],
            cast_slice(self.data.as_slice()),
        )?;
        let serialized = serialize(
            std::iter::once((Self::TENSOR_NAME.to_string(), tensor_view)),
            &None,
        )?;
        let mut file = File::create(path)?;
        file.write_all(&serialized)?;
        Ok(())
    }

    /// Reads a snapshot from a `.stt` safetensors file.
    pub fn read<P: AsRef<Path>>(path: P) -> Result<Self, EmbeddingSnapshotError> {
        let path = path.as_ref();
        let mut bytes = Vec::new();
        File::open(path)?.read_to_end(&mut bytes)?;
        let tensors = SafeTensors::deserialize(&bytes)?;
        let tensor = tensors.tensor(Self::TENSOR_NAME)?;
        if tensor.dtype() != safetensors::Dtype::F32 {
            return Err(EmbeddingSnapshotError::InvalidDType(tensor.dtype()));
        }
        let shape = tensor.shape();
        if shape.len() != 2 {
            return Err(EmbeddingSnapshotError::InvalidShape);
        }
        let rows = shape[0];
        let dims = shape[1];
        let data = cast_slice::<u8, f32>(tensor.data()).to_vec();
        Ok(Self { rows, dims, data })
    }

    /// Returns the number of embeddings stored in the snapshot.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the embedding dimensionality.
    pub fn dims(&self) -> usize {
        self.dims
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn snapshot_round_trip_preserves_tensor() {
        let tensor =
            Tensor::from_vec(vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0], (2, 3), &Device::Cpu).unwrap();
        let snapshot = EmbeddingSnapshot::from_tensor(&tensor).unwrap();
        let file = NamedTempFile::new().unwrap();
        snapshot.write(file.path()).unwrap();
        let restored = EmbeddingSnapshot::read(file.path()).unwrap();
        assert_eq!(snapshot, restored);
        let tensor_restored = restored.to_tensor(&Device::Cpu).unwrap();
        assert_eq!(
            tensor_restored.to_vec2::<f32>().unwrap(),
            tensor.to_vec2::<f32>().unwrap()
        );
        assert_eq!(restored.rows(), 2);
        assert_eq!(restored.dims(), 3);
    }
}
