use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct TableEncoder {
    table: HashMap<String, Vec<f32>>,
    dimension: usize,
}

impl TableEncoder {
    pub fn new(table: HashMap<String, Vec<f32>>) -> Self {
        let dimension = table.values().next().map(|v| v.len()).unwrap_or(0);
        Self { table, dimension }
    }

    pub fn encode(&self, input: &str) -> Vec<f32> {
        self.table
            .get(input)
            .cloned()
            .unwrap_or_else(|| vec![0.0; self.dimension])
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BinaryDecoder {
    threshold: f32,
}

impl BinaryDecoder {
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }

    pub fn decode(&self, value: f32) -> String {
        if value > self.threshold {
            "1".to_string()
        } else {
            "0".to_string()
        }
    }
}
