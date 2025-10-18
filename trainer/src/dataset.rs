use memmap2::Mmap;
use serde::Deserialize;
use std::{fs::File, path::Path};

#[derive(Debug, Deserialize, Clone, PartialEq)]
pub struct DecoderSample {
    pub context: String,
    pub target_tokens: Vec<String>,
    pub target_embedding: Vec<f32>,
    pub retrieval_candidates: Vec<Vec<f32>>,
}

pub struct MmapDataset {
    mmap: Mmap,
}

impl MmapDataset {
    pub fn open(path: &Path) -> anyhow::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(Self { mmap })
    }

    pub fn iter(&self) -> impl Iterator<Item = anyhow::Result<DecoderSample>> + '_ {
        std::str::from_utf8(&self.mmap)
            .expect("utf8 dataset")
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| serde_json::from_str(line).map_err(anyhow::Error::from))
    }
}
