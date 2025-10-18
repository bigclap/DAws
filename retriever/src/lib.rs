//! Placeholder ANN retriever module for the multi-crate workspace skeleton.

pub mod io;

pub use io::{KvMemorySnapshot, MemoryMetadata, MemorySnapshotError};

/// Minimal configuration stub so downstream crates can be wired incrementally.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RetrieverConfig {
    /// Number of neighbours to query from the backing index.
    pub top_k: usize,
}

/// No-op retriever implementation that records configuration.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Retriever {
    config: RetrieverConfig,
}

impl Retriever {
    /// Creates a retriever with the provided [`RetrieverConfig`].
    pub fn new(config: RetrieverConfig) -> Self {
        Self { config }
    }

    /// Returns the configured `top_k` parameter.
    pub fn top_k(&self) -> usize {
        self.config.top_k
    }
}

#[cfg(test)]
mod tests {
    use super::{Retriever, RetrieverConfig};

    #[test]
    fn configuration_round_trip() {
        let retriever = Retriever::new(RetrieverConfig { top_k: 42 });
        assert_eq!(retriever.top_k(), 42);
    }
}
