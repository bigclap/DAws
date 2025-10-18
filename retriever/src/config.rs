use std::fmt;

/// Configuration for a [`Retriever`](crate::Retriever) instance.
#[derive(Clone, Debug, PartialEq)]
pub struct RetrieverConfig {
    /// Dimensionality of the embedding vectors stored in the index.
    pub dimension: usize,
    /// Dimensionality of the value vectors returned for each key.
    pub value_dimension: usize,
    /// Maximum number of elements that may be inserted into the index.
    pub max_elements: usize,
    /// Maximum number of HNSW layers retained in the structure.
    pub max_layers: usize,
    /// Maximum number of neighbour connections maintained per node.
    pub max_connections: usize,
    /// Construction time search depth for HNSW insertions.
    pub ef_construction: usize,
    /// Search time beam width for ANN queries.
    pub ef_search: usize,
    /// Default number of neighbours returned when querying.
    pub top_k: usize,
}

impl Default for RetrieverConfig {
    fn default() -> Self {
        Self {
            dimension: 0,
            value_dimension: 0,
            max_elements: 0,
            max_layers: 16,
            max_connections: 16,
            ef_construction: 200,
            ef_search: 50,
            top_k: 8,
        }
    }
}

/// Key-value record stored within the persistent memory.
#[derive(Clone, Debug, PartialEq)]
pub struct MemoryRecord {
    /// Unique identifier for the memory entry.
    pub key: u64,
    /// Embedding used for ANN search.
    pub embedding: Vec<f32>,
    /// Value returned for diffusion recruitment.
    pub value: Vec<f32>,
}

impl MemoryRecord {
    /// Creates a new [`MemoryRecord`].
    pub fn new(key: u64, embedding: Vec<f32>, value: Vec<f32>) -> Self {
        Self {
            key,
            embedding,
            value,
        }
    }
}

/// Result of querying the retriever for approximate neighbours.
#[derive(Clone, Debug, PartialEq)]
pub struct MemoryHit {
    /// Key associated with the retrieved memory.
    pub key: u64,
    /// Cosine similarity between query and stored embedding.
    pub similarity: f32,
}

impl fmt::Display for MemoryHit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({:.4})", self.key, self.similarity)
    }
}
