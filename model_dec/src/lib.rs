//! Decoder utilities translating activations into symbolic outputs.

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

#[cfg(test)]
mod tests {
    use super::BinaryDecoder;

    #[test]
    fn thresholding_behaviour_matches_expectations() {
        let decoder = BinaryDecoder::new(0.4);
        assert_eq!(decoder.decode(0.3), "0");
        assert_eq!(decoder.decode(0.5), "1");
    }
}
