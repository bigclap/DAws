use std::io::{Read, Write};
use std::path::PathBuf;

use bytemuck::cast_slice;
use safetensors::tensor::TensorView;
use safetensors::{SafeTensors, serialize};

use super::MemorySnapshotError;

pub fn write_f32_matrix(
    path: PathBuf,
    name: &str,
    data: &[f32],
    rows: usize,
    dims: usize,
) -> Result<(), MemorySnapshotError> {
    let expected = rows
        .checked_mul(dims)
        .ok_or_else(|| MemorySnapshotError::InvalidShape {
            expected: rows,
            found: dims,
        })?;
    if expected != data.len() {
        return Err(MemorySnapshotError::InvalidShape {
            expected,
            found: data.len(),
        });
    }
    let tensor = TensorView::new(safetensors::Dtype::F32, vec![rows, dims], cast_slice(data))?;
    let bytes = serialize(std::iter::once((name.to_string(), tensor)), &None)?;
    std::fs::File::create(path)?.write_all(&bytes)?;
    Ok(())
}

pub fn read_f32_matrix(
    path: PathBuf,
    name: &str,
) -> Result<(Vec<f32>, usize, usize), MemorySnapshotError> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)?.read_to_end(&mut bytes)?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let tensor = tensors.tensor(name)?;
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

pub fn write_f32_vector(
    path: PathBuf,
    name: &str,
    data: &[f32],
) -> Result<(), MemorySnapshotError> {
    let tensor = TensorView::new(safetensors::Dtype::F32, vec![data.len()], cast_slice(data))?;
    let bytes = serialize(std::iter::once((name.to_string(), tensor)), &None)?;
    std::fs::File::create(path)?.write_all(&bytes)?;
    Ok(())
}

pub fn read_f32_vector(
    path: PathBuf,
    name: &str,
) -> Result<(Vec<f32>, usize), MemorySnapshotError> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)?.read_to_end(&mut bytes)?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let tensor = tensors.tensor(name)?;
    if tensor.dtype() != safetensors::Dtype::F32 {
        return Err(MemorySnapshotError::UnsupportedDType {
            dtype: tensor.dtype(),
        });
    }
    let shape = tensor.shape();
    if shape.len() != 1 {
        return Err(MemorySnapshotError::InvalidShape {
            expected: 1,
            found: shape.len(),
        });
    }
    let len = shape[0];
    let data = cast_slice::<u8, f32>(tensor.data()).to_vec();
    Ok((data, len))
}

pub fn write_keys(path: PathBuf, name: &str, keys: &[u64]) -> Result<(), MemorySnapshotError> {
    let casted: Vec<i64> = keys
        .iter()
        .map(|&key| {
            i64::try_from(key).map_err(|_| MemorySnapshotError::MetadataOverflow {
                value: usize::try_from(key).unwrap_or(usize::MAX),
            })
        })
        .collect::<Result<_, _>>()?;
    let tensor = TensorView::new(
        safetensors::Dtype::I64,
        vec![casted.len()],
        cast_slice(&casted),
    )?;
    let bytes = serialize(std::iter::once((name.to_string(), tensor)), &None)?;
    std::fs::File::create(path)?.write_all(&bytes)?;
    Ok(())
}

pub fn read_keys(path: PathBuf, name: &str) -> Result<Vec<u64>, MemorySnapshotError> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)?.read_to_end(&mut bytes)?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let tensor = tensors.tensor(name)?;
    if tensor.dtype() != safetensors::Dtype::I64 {
        return Err(MemorySnapshotError::UnsupportedDType {
            dtype: tensor.dtype(),
        });
    }
    let shape = tensor.shape();
    if shape.len() != 1 {
        return Err(MemorySnapshotError::InvalidShape {
            expected: 1,
            found: shape.len(),
        });
    }
    let data = cast_slice::<u8, i64>(tensor.data())
        .iter()
        .map(|&value| u64::try_from(value).map_err(|_| MemorySnapshotError::InvalidMetadata(value)))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(data)
}
