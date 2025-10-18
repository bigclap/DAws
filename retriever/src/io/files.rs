use std::fs::File;
use std::io::{Read, Write};
use std::path::PathBuf;

use super::MemorySnapshotError;

pub fn write_index(path: PathBuf, bytes: &[u8]) -> Result<(), MemorySnapshotError> {
    let mut file = File::create(path)?;
    file.write_all(bytes)?;
    Ok(())
}

pub fn read_index(path: PathBuf) -> Result<Vec<u8>, MemorySnapshotError> {
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;
    Ok(bytes)
}
