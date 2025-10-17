//! Minimal embedding encoder/decoder utilities used by the examples.

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

#[derive(Clone, Copy, Debug)]
/// Binary threshold decoder translating scalar activations to "0" or "1".
pub struct BinaryDecoder {
    threshold: f32,
}

impl BinaryDecoder {
    /// Constructs a decoder that emits `"1"` when the value exceeds `threshold`.
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }

    /// Converts a scalar activation into a string class label.
    pub fn decode(&self, value: f32) -> String {
        if value > self.threshold {
            "1".to_string()
        } else {
            "0".to_string()
        }
    }
}
