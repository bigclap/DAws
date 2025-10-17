//! Minimal embedding encoder utilities used by the examples.

use std::collections::HashMap;

#[derive(Clone, Debug)]
/// Lookup-based encoder that maps discrete tokens to dense vectors.
pub struct TableEncoder {
    table: HashMap<String, Vec<f32>>,
    dimension: usize,
}

impl TableEncoder {
    /// Creates a new encoder from a provided token → vector mapping.
    pub fn new(table: HashMap<String, Vec<f32>>) -> Self {
        let dimension = table.values().next().map(|v| v.len()).unwrap_or(0);
        Self { table, dimension }
    }

    /// Returns the embedding for `input` or zeros if it is unseen.
    pub fn encode(&self, input: &str) -> Vec<f32> {
        self.table
            .get(input)
            .cloned()
            .unwrap_or_else(|| vec![0.0; self.dimension])
    }

    /// Embedding dimensionality inferred from the lookup table.
    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

#[cfg(test)]
mod tests {
    use super::TableEncoder;
    use std::collections::HashMap;

    #[test]
    fn unseen_token_returns_zeros() {
        let mut table = HashMap::new();
        table.insert("hello".to_string(), vec![1.0, 0.5]);
        let encoder = TableEncoder::new(table);

        assert_eq!(encoder.dimension(), 2);
        assert_eq!(encoder.encode("missing"), vec![0.0, 0.0]);
    }
}
