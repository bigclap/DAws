use std::sync::Arc;

use parquet::data_type::Int64Type;
use parquet::file::properties::WriterProperties;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::file::writer::SerializedFileWriter;
use parquet::record::RowAccessor;
use parquet::schema::parser::parse_message_type;

use super::MemorySnapshotError;

/// Metadata stored alongside the ANN memory vectors.
#[derive(Clone, Debug, PartialEq)]
pub struct MemoryMetadata {
    /// Number of vector entries persisted in the shard.
    pub num_vectors: usize,
    /// Dimensionality of each vector.
    pub dimension: usize,
    /// Dimensionality of the stored value vectors.
    pub value_dimension: usize,
}

pub fn write_metadata(
    path: std::path::PathBuf,
    metadata: &MemoryMetadata,
) -> Result<(), MemorySnapshotError> {
    let schema = Arc::new(parse_message_type(
        "message kv_meta {\n  REQUIRED INT64 num_vectors;\n  REQUIRED INT64 dimension;\n  REQUIRED INT64 value_dimension;\n}",
    )?);
    let props = Arc::new(WriterProperties::builder().build());
    let file = std::fs::File::create(path)?;
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
        if let Some(mut col) = row_group.next_column()? {
            let writer = col.typed::<Int64Type>();
            let values = [usize_to_i64(metadata.value_dimension)?];
            writer.write_batch(&values, None, None)?;
            col.close()?;
        }
        row_group.close()?;
    }
    writer.close()?;
    Ok(())
}

pub fn read_metadata(path: std::path::PathBuf) -> Result<MemoryMetadata, MemorySnapshotError> {
    let file = std::fs::File::open(path)?;
    let reader = SerializedFileReader::new(file)?;
    let mut rows = reader.get_row_iter(None)?;
    let row = match rows.next() {
        Some(result) => result?,
        None => return Err(MemorySnapshotError::EmptyMetadata),
    };
    let num_vectors = row.get_long(0)?;
    let dimension = row.get_long(1)?;
    let value_dimension = row.get_long(2)?;
    if num_vectors < 0 {
        return Err(MemorySnapshotError::InvalidMetadata(num_vectors));
    }
    if dimension < 0 {
        return Err(MemorySnapshotError::InvalidMetadata(dimension));
    }
    if value_dimension < 0 {
        return Err(MemorySnapshotError::InvalidMetadata(value_dimension));
    }
    Ok(MemoryMetadata {
        num_vectors: usize::try_from(num_vectors)
            .map_err(|_| MemorySnapshotError::InvalidMetadata(num_vectors))?,
        dimension: usize::try_from(dimension)
            .map_err(|_| MemorySnapshotError::InvalidMetadata(dimension))?,
        value_dimension: usize::try_from(value_dimension)
            .map_err(|_| MemorySnapshotError::InvalidMetadata(value_dimension))?,
    })
}

fn usize_to_i64(value: usize) -> Result<i64, MemorySnapshotError> {
    if value > i64::MAX as usize {
        Err(MemorySnapshotError::MetadataOverflow { value })
    } else {
        Ok(value as i64)
    }
}
