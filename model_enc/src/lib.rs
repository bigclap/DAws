//! Candle-backed text encoder utilities used by the diffusion stack.

use std::fs;

use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::{Module, VarBuilder, embedding::Embedding};
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Repo, RepoType};
use thiserror::Error;
use tokenizers::Tokenizer;

/// Errors that may occur while loading or running the encoder.
#[derive(Debug, Error)]
pub enum EncoderError {
    /// Raised when the Hugging Face hub API fails.
    #[error("hf-hub API error: {0}")]
    Hub(#[from] hf_hub::api::sync::ApiError),
    /// Raised when reading configuration files fails.
    #[error("failed to read config file: {0}")]
    Io(#[from] std::io::Error),
    /// Raised when JSON parsing fails.
    #[error("failed to parse config: {0}")]
    Parse(#[from] serde_json::Error),
    /// Raised when tokenization fails.
    #[error("tokenization failed: {0}")]
    Tokenizer(#[from] Box<dyn std::error::Error + Send + Sync>),
    /// Raised when candle operations fail.
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
    /// Raised when configuration is invalid.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
}

/// A frozen encoder that can be backed either by a Hugging Face BERT checkpoint or by a
/// lightweight embedding table used for tests.
pub struct FrozenTextEncoder {
    inner: EncoderInner,
    device: Device,
    pad_token_id: u32,
    hidden_size: usize,
}

enum EncoderInner {
    Bert {
        model: BertModel,
        tokenizer: Tokenizer,
    },
    Embedding {
        embedding: Embedding,
        tokenizer: Tokenizer,
    },
}

impl FrozenTextEncoder {
    /// Loads a sentence encoder checkpoint from the Hugging Face hub.
    ///
    /// The checkpoint must expose a BERT compatible `config.json`, `tokenizer.json`, and
    /// `model.safetensors` file. This covers the `e5-small-v2`, `all-MiniLM-L6-v2`, and similar
    /// text encoders built on top of BERT.
    pub fn from_hf_repo(
        repo_id: &str,
        revision: Option<&str>,
        device: Device,
    ) -> Result<Self, EncoderError> {
        let api = ApiBuilder::new().build()?;
        let repo = if let Some(revision) = revision {
            Repo::with_revision(repo_id.to_string(), RepoType::Model, revision.to_string())
        } else {
            Repo::new(repo_id.to_string(), RepoType::Model)
        };
        let repo = api.repo(repo);
        let tokenizer_path = repo.get("tokenizer.json")?;
        let config_path = repo.get("config.json")?;
        let weights_path = repo.get("model.safetensors")?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(EncoderError::Tokenizer)?;
        let config: BertConfig = serde_json::from_str(&fs::read_to_string(&config_path)?)?;
        let pad_token_id = config.pad_token_id as u32;

        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)? };
        let model = BertModel::load(vb, &config)?;

        Ok(Self {
            inner: EncoderInner::Bert { model, tokenizer },
            pad_token_id,
            hidden_size: config.hidden_size,
            device,
        })
    }

    /// Constructs the encoder from a static embedding table. This is primarily used by unit tests
    /// to validate pooling behaviour without downloading external checkpoints.
    pub fn from_static_embeddings(
        tokenizer: Tokenizer,
        embedding_table: Tensor,
        pad_token_id: u32,
    ) -> Result<Self, EncoderError> {
        let hidden_size = embedding_table.dim(D::Minus1)?;
        let device = embedding_table.device().clone();
        let embedding = Embedding::new(embedding_table, hidden_size);
        Ok(Self {
            inner: EncoderInner::Embedding {
                embedding,
                tokenizer,
            },
            pad_token_id,
            hidden_size,
            device,
        })
    }

    /// Embedding dimensionality.
    pub fn dimension(&self) -> usize {
        self.hidden_size
    }

    /// Encodes a single string, returning a pooled embedding tensor of shape `(hidden_size,)`.
    pub fn encode(&self, text: &str) -> Result<Tensor, EncoderError> {
        let batch = self.encode_batch(&[text])?;
        Ok(batch.i(0)?.squeeze(0)?)
    }

    /// Encodes a batch of strings returning a tensor with shape `(batch, hidden_size)`.
    pub fn encode_batch<T>(&self, texts: &[T]) -> Result<Tensor, EncoderError>
    where
        T: AsRef<str>,
    {
        if texts.is_empty() {
            return Ok(Tensor::zeros(
                (0, self.hidden_size),
                DType::F32,
                &self.device,
            )?);
        }

        let tokenizer = match &self.inner {
            EncoderInner::Bert { tokenizer, .. } => tokenizer,
            EncoderInner::Embedding { tokenizer, .. } => tokenizer,
        };

        let encodings = tokenizer
            .encode_batch(texts.iter().map(AsRef::as_ref).collect::<Vec<_>>(), true)
            .map_err(EncoderError::Tokenizer)?;

        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0);

        if max_len == 0 {
            return Ok(Tensor::zeros(
                (texts.len(), self.hidden_size),
                DType::F32,
                &self.device,
            )?);
        }

        let pad = self.pad_token_id as i64;
        let batch = encodings.len();
        let mut input_ids = Vec::with_capacity(batch * max_len);
        let mut attention = Vec::with_capacity(batch * max_len);

        for encoding in encodings {
            let mut ids = encoding
                .get_ids()
                .iter()
                .map(|id| *id as i64)
                .collect::<Vec<_>>();
            let mut mask = encoding
                .get_attention_mask()
                .iter()
                .map(|m| *m as i64)
                .collect::<Vec<_>>();

            while ids.len() < max_len {
                ids.push(pad);
                mask.push(0);
            }

            input_ids.extend_from_slice(&ids);
            attention.extend_from_slice(&mask);
        }

        let ids_tensor = Tensor::from_vec(input_ids, (batch, max_len), &self.device)?;
        let mask_tensor = Tensor::from_vec(attention, (batch, max_len), &self.device)?;

        let embeddings = match &self.inner {
            EncoderInner::Bert { model, .. } => {
                let token_types = Tensor::zeros((batch, max_len), DType::I64, &self.device)?;
                model.forward(&ids_tensor, &token_types, Some(&mask_tensor))?
            }
            EncoderInner::Embedding { embedding, .. } => embedding.forward(&ids_tensor)?,
        };

        let mask = mask_tensor.to_dtype(DType::F32)?;
        let mask = mask.unsqueeze(2)?;
        let masked_embeddings = embeddings.broadcast_mul(&mask)?;
        let summed = masked_embeddings.sum(1)?;
        let counts = mask.sum(1)?;
        let epsilon = Tensor::full(1e-6f32, counts.shape(), &self.device)?;
        let counts = counts.broadcast_add(&epsilon)?;
        let pooled = summed.broadcast_div(&counts)?;
        Ok(pooled)
    }
}

#[cfg(test)]
mod tests {
    use super::FrozenTextEncoder;
    use candle_core::{Device, Tensor};
    use std::collections::HashMap;
    use tokenizers::Tokenizer;
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;
    use tokenizers::processors::template::TemplateProcessing;
    use tokenizers::utils::padding::PaddingParams;

    fn build_tokenizer() -> Tokenizer {
        let mut vocab = HashMap::new();
        vocab.insert("[PAD]".to_string(), 0);
        vocab.insert("[UNK]".to_string(), 1);
        vocab.insert("[CLS]".to_string(), 2);
        vocab.insert("[SEP]".to_string(), 3);
        vocab.insert("hello".to_string(), 4);
        vocab.insert("world".to_string(), 5);

        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap();

        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Whitespace::default());
        let processor = TemplateProcessing::builder()
            .try_single(vec!["[CLS]", "$0", "[SEP]"])
            .unwrap()
            .special_tokens(vec![("[CLS]", 2), ("[SEP]", 3)])
            .build()
            .unwrap();
        tokenizer.with_post_processor(processor);
        tokenizer.with_padding(Some(PaddingParams::default()));
        tokenizer
    }

    #[test]
    fn mean_pooling_matches_expected_static_embeddings() {
        let tokenizer = build_tokenizer();
        let weights = Tensor::from_vec(
            vec![
                0.0f32, 0.0, 0.0, 0.0, // [PAD]
                0.1, 0.1, 0.1, 0.1, // [UNK]
                0.2, 0.0, 0.0, 0.2, // [CLS]
                0.0, 0.2, 0.2, 0.0, // [SEP]
                1.0, 0.0, 0.0, 0.0, // hello
                0.0, 1.0, 0.0, 0.0, // world
            ],
            (6, 4),
            &Device::Cpu,
        )
        .unwrap();

        let encoder = FrozenTextEncoder::from_static_embeddings(tokenizer, weights, 0).unwrap();
        let embedding = encoder.encode("hello world").unwrap();
        let values = embedding.to_vec1::<f32>().unwrap();
        assert_eq!(encoder.dimension(), 4);
        let expected = vec![0.3, 0.3, 0.05, 0.05];
        for (value, expected) in values.iter().zip(expected.iter()) {
            assert!((value - expected).abs() < 1e-6);
        }
    }
}
