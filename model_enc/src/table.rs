//! Simple table-based encoder used for deterministic evaluation fixtures.

use std::collections::HashMap;

/// Lookup encoder that maps string keys to fixed embeddings.
#[derive(Clone, Debug)]
pub struct TableEncoder {
    table: HashMap<String, Vec<f32>>,
    dimension: usize,
}

impl TableEncoder {
    /// Creates a table encoder ensuring all embeddings share the same dimensionality.
    pub fn new(table: HashMap<String, Vec<f32>>) -> Self {
        let dimension = table.values().next().map(|v| v.len()).unwrap_or(0);
        for (key, values) in &table {
            assert_eq!(
                values.len(),
                dimension,
                "embedding dimension mismatch for key {key}",
            );
        }
        Self { table, dimension }
    }

    /// Returns the dimensionality of the embeddings stored in the table.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Encodes the provided key, yielding the corresponding embedding.
    ///
    /// When the key is absent, a zero vector with the configured dimensionality is returned.
    pub fn encode(&self, key: &str) -> Vec<f32> {
        self.table
            .get(key)
            .cloned()
            .unwrap_or_else(|| vec![0.0; self.dimension])
    }
}

#[cfg(test)]
mod tests {
    use super::TableEncoder;
    use std::collections::HashMap;

    #[test]
    fn encoder_returns_embeddings_for_known_keys() {
        let mut table = HashMap::new();
        table.insert("a".to_string(), vec![1.0, 0.0]);
        table.insert("b".to_string(), vec![0.0, 1.0]);
        let encoder = TableEncoder::new(table);
        assert_eq!(encoder.dimension(), 2);
        assert_eq!(encoder.encode("a"), vec![1.0, 0.0]);
        assert_eq!(encoder.encode("b"), vec![0.0, 1.0]);
        assert_eq!(encoder.encode("missing"), vec![0.0, 0.0]);
    }

    #[test]
    #[should_panic(expected = "embedding dimension mismatch")]
    fn inconsistent_dimensions_panic() {
        let mut table = HashMap::new();
        table.insert("a".to_string(), vec![1.0]);
        table.insert("b".to_string(), vec![0.0, 1.0]);
        let _ = TableEncoder::new(table);
    }
}
