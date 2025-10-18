//! Decoder utilities translating encoder embeddings into cosine-aligned targets.

use candle_core::{D, Device, Result as CandleResult, Tensor};
use candle_nn::{Linear, Module, RNN, VarBuilder, linear, ops, rnn::gru};
use thiserror::Error;

/// Errors emitted by the decoder when parameter loading or inference fails.
#[derive(Debug, Error)]
pub enum DecoderError {
    /// Raised when Candle reports an error.
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
    /// Raised when the configuration is inconsistent.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
}

/// Configuration for the [`TransformerGruDecoder`].
#[derive(Debug, Clone)]
pub struct TransformerGruDecoderConfig {
    /// Hidden size of the decoder representations.
    pub hidden_size: usize,
    /// Number of attention heads.
    pub num_attention_heads: usize,
    /// Number of virtual prefix tokens prepended to the context.
    pub prefix_tokens: usize,
    /// Dropout probability used inside the attention block (currently unused but reserved).
    pub dropout_prob: f64,
}

impl TransformerGruDecoderConfig {
    /// Verifies that the configuration is internally consistent.
    fn validate(&self) -> Result<(), DecoderError> {
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(DecoderError::InvalidConfig(
                "hidden size must be divisible by the number of heads".to_string(),
            ));
        }
        if self.prefix_tokens == 0 {
            return Err(DecoderError::InvalidConfig(
                "at least one prefix token is required".to_string(),
            ));
        }
        Ok(())
    }
}

/// Multi-head self-attention used inside the decoder.
#[derive(Clone)]
struct MultiHeadSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl MultiHeadSelfAttention {
    fn load(config: &TransformerGruDecoderConfig, vb: VarBuilder) -> CandleResult<Self> {
        let head_dim = config.hidden_size / config.num_attention_heads;
        let q_proj = linear(config.hidden_size, config.hidden_size, vb.pp("q_proj"))?;
        let k_proj = linear(config.hidden_size, config.hidden_size, vb.pp("k_proj"))?;
        let v_proj = linear(config.hidden_size, config.hidden_size, vb.pp("v_proj"))?;
        let out_proj = linear(config.hidden_size, config.hidden_size, vb.pp("out_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads: config.num_attention_heads,
            head_dim,
            scale: (head_dim as f64).sqrt().recip(),
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let batch = x.dim(0)?;
        let seq = x.dim(1)?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((batch, seq, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch, seq, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch, seq, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        let scores = q.matmul(&k.transpose(2, 3)?)?;
        let scores = (scores * self.scale)?;
        let attn = ops::softmax(&scores, D::Minus1)?;
        let context = attn.matmul(&v)?;
        let context =
            context
                .transpose(1, 2)?
                .reshape((batch, seq, self.num_heads * self.head_dim))?;
        self.out_proj.forward(&context)
    }
}

/// Decoder that combines a lightweight Transformer block with a GRU tail.
#[derive(Clone)]
pub struct TransformerGruDecoder {
    device: Device,
    hidden_size: usize,
    prefix: Tensor,
    attention: MultiHeadSelfAttention,
    gru: candle_nn::rnn::GRU,
    projection: Linear,
}

impl TransformerGruDecoder {
    /// Loads the decoder parameters from a [`VarBuilder`].
    pub fn load(config: TransformerGruDecoderConfig, vb: VarBuilder) -> Result<Self, DecoderError> {
        config.validate()?;
        let prefix = vb.get((config.prefix_tokens, config.hidden_size), "prefix")?;
        let attention = MultiHeadSelfAttention::load(&config, vb.pp("attention"))?;
        let gru = gru(
            config.hidden_size,
            config.hidden_size,
            candle_nn::rnn::GRUConfig::default(),
            vb.pp("gru"),
        )?;
        let projection = linear(config.hidden_size, config.hidden_size, vb.pp("projection"))?;
        let device = vb.device().clone();
        Ok(Self {
            device,
            hidden_size: config.hidden_size,
            prefix,
            attention,
            gru,
            projection,
        })
    }

    /// Applies the decoder to a batch of context embeddings.
    ///
    /// `context` must have shape `(batch, steps, hidden_size)` and be located on the same device as
    /// the decoder parameters.
    pub fn decode(&self, context: &Tensor) -> Result<Tensor, DecoderError> {
        let batch = context.dim(0)?;
        let prefix_len = self.prefix.dim(0)?;
        let prefix = self
            .prefix
            .unsqueeze(0)?
            .expand((batch, prefix_len, self.hidden_size))?;
        let context = context.to_device(&self.device)?;
        let tokens = Tensor::cat(&[prefix, context], 1)?;
        let attended = self.attention.forward(&tokens)?;
        let tokens = (tokens + attended)?;
        let states = self.gru.seq(&tokens)?;
        let final_state = states
            .last()
            .ok_or_else(|| DecoderError::InvalidConfig("empty sequence".into()))?
            .h
            .clone();
        Ok(self.projection.forward(&final_state)?)
    }
}

/// Computes the cosine alignment loss between predictions and targets.
pub fn cosine_alignment_loss(predictions: &Tensor, targets: &Tensor) -> CandleResult<Tensor> {
    let predictions = safe_normalise(predictions)?;
    let targets = safe_normalise(targets)?;
    let cosine = (predictions * targets)?.sum(D::Minus1)?;
    let cosine = cosine.clamp(-1.0, 1.0)?;
    let losses = (1.0f64 - cosine)?;
    losses.mean(0)
}

fn safe_normalise(x: &Tensor) -> CandleResult<Tensor> {
    let norm = x.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    let epsilon = Tensor::full(1e-6f32, norm.shape(), x.device())?;
    let denom = norm.broadcast_add(&epsilon)?;
    x.broadcast_div(&denom)
}

#[cfg(test)]
mod tests {
    use super::{TransformerGruDecoder, TransformerGruDecoderConfig, cosine_alignment_loss};
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;
    use std::collections::HashMap;

    fn build_decoder() -> TransformerGruDecoder {
        let device = Device::Cpu;
        let mut tensors = HashMap::new();
        tensors.insert(
            "prefix".to_string(),
            Tensor::from_vec(vec![0.0f32, 0.0], (1, 2), &device).unwrap(),
        );
        for name in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            tensors.insert(
                format!("attention.{name}.weight"),
                Tensor::from_vec(vec![0.0f32; 4], (2, 2), &device).unwrap(),
            );
            tensors.insert(
                format!("attention.{name}.bias"),
                Tensor::from_vec(vec![0.0f32; 2], (2,), &device).unwrap(),
            );
        }

        let mut gru_bias = vec![0.0f32; 6];
        gru_bias[4] = 1.0;
        gru_bias[5] = 1.0;
        tensors.insert(
            "gru.bias_ih_l0".to_string(),
            Tensor::from_vec(gru_bias, (6,), &device).unwrap(),
        );
        tensors.insert(
            "gru.bias_hh_l0".to_string(),
            Tensor::from_vec(vec![0.0f32; 6], (6,), &device).unwrap(),
        );
        tensors.insert(
            "gru.weight_ih_l0".to_string(),
            Tensor::from_vec(vec![0.0f32; 12], (6, 2), &device).unwrap(),
        );
        tensors.insert(
            "gru.weight_hh_l0".to_string(),
            Tensor::from_vec(vec![0.0f32; 12], (6, 2), &device).unwrap(),
        );

        tensors.insert(
            "projection.weight".to_string(),
            Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (2, 2), &device).unwrap(),
        );
        tensors.insert(
            "projection.bias".to_string(),
            Tensor::from_vec(vec![0.0f32; 2], (2,), &device).unwrap(),
        );

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let config = TransformerGruDecoderConfig {
            hidden_size: 2,
            num_attention_heads: 1,
            prefix_tokens: 1,
            dropout_prob: 0.0,
        };
        TransformerGruDecoder::load(config, vb).unwrap()
    }

    #[test]
    fn decode_respects_prefix_bias() {
        let decoder = build_decoder();
        let context = Tensor::from_vec(vec![0.2f32, -0.1], (1, 1, 2), &Device::Cpu).unwrap();
        let decoded = decoder.decode(&context).unwrap();
        let values = decoded.to_vec2::<f32>().unwrap();
        assert_eq!(values.len(), 1);
        assert_eq!(values[0].len(), 2);
        let expected = 0.5711956; // Derived from the GRU bias influence with zeroed weights.
        assert!((values[0][0] - expected).abs() < 1e-5);
        assert!((values[0][1] - expected).abs() < 1e-5);
    }

    #[test]
    fn cosine_loss_handles_degenerate_vectors() {
        let preds = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 0.0], (2, 2), &Device::Cpu).unwrap();
        let targets = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (2, 2), &Device::Cpu).unwrap();
        let loss = cosine_alignment_loss(&preds, &targets).unwrap();
        let value = loss.to_vec0::<f32>().unwrap();
        assert!((value - 0.5).abs() < 1e-6);
    }
}
