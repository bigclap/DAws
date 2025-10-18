//! Utility decoder that thresholds activations into binary string outputs.

/// Decoder that maps scalar activations into binary tokens using a configurable threshold.
#[derive(Clone, Debug)]
pub struct BinaryDecoder {
    threshold: f32,
}

impl BinaryDecoder {
    /// Creates a decoder that emits "1" when the activation meets the threshold and "0" otherwise.
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }

    /// Applies the decoder to the supplied activation value.
    pub fn decode(&self, activation: f32) -> String {
        if activation >= self.threshold {
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
    fn thresholding_behaves_as_expected() {
        let decoder = BinaryDecoder::new(0.5);
        assert_eq!(decoder.decode(0.75), "1");
        assert_eq!(decoder.decode(0.5), "1");
        assert_eq!(decoder.decode(0.49), "0");
    }
}
