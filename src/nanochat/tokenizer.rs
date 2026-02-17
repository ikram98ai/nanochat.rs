use candle_core::{Result};
use std::path::{Path};
use tokenizers::Tokenizer as TokenizerInternal;

#[allow(dead_code)]
pub enum ContentPart {
    Text(String),
    Image(String),
}

#[allow(dead_code)]
pub struct Message {
    pub role: String,
    pub content: MessageContent,
}

#[allow(dead_code)]
pub enum MessageContent {
    Simple(String),
    Parts(Vec<ContentPart>),
}

#[allow(dead_code)]
pub struct Conversation {
    pub messages: Vec<Message>,
    pub tools: Option<Vec<String>>,
}

#[derive(Clone)]
pub struct Tokenizer {
    inner: TokenizerInternal,
}

impl Tokenizer {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let inner = TokenizerInternal::from_file(path)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        Ok(Self { inner })
    }

    pub fn encode(&self, text: &str, add_special: bool) -> Result<Vec<u32>> {
        let encoding = self.inner.encode(text, add_special)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn encode_special(&self, text: &str) -> u32 {
        self.inner.token_to_id(text).unwrap_or(0)
    }

    #[allow(dead_code)]
    pub fn encode_with_special(&self, text: &str, _prepend: Option<u32>, _append: Option<u32>) -> Result<Vec<u32>> {
        self.encode(text, false)
    }

    #[allow(dead_code)]
    pub fn encode_batch(&self, texts: &[&str], _prepend: Option<u32>, _append: Option<u32>) -> Result<Vec<Vec<u32>>> {
        let mut results = Vec::new();
        for text in texts {
            results.push(self.encode(text, false)?);
        }
        Ok(results)
    }

    #[allow(dead_code)]
    pub fn get_vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    #[allow(dead_code)]
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> Result<String> {
        self.inner.decode(ids, skip_special)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))
    }

    pub fn get_bos_token_id(&self) -> u32 {
        self.inner.token_to_id("<|bos|>").unwrap_or(0)
    }

    #[allow(dead_code)]
    pub fn get_special_tokens(&self) -> Vec<String> {
        vec![]
    }

    #[allow(dead_code)]
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        self.inner.save(path.as_ref(), true)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        Ok(())
    }

    #[allow(dead_code)]
    pub fn render_conversation(&self, _conv: &Conversation, _max_tokens: usize) -> Result<(Vec<u32>, Vec<u8>)> {
        Ok((vec![], vec![]))
    }

    #[allow(dead_code)]
    pub fn render_for_completion(&self, _conv: &Conversation) -> Result<Vec<u32>> {
        Ok(vec![])
    }

    #[allow(dead_code)]
    pub fn visualize_tokenization(&self, _ids: &[u32], _mask: &[u8]) -> Result<String> {
        Ok(String::new())
    }
}

#[allow(dead_code)]
pub fn get_tokenizer() -> Result<Tokenizer> {
    let path = crate::nanochat::common::get_base_dir().join("tokenizer.json");
    Tokenizer::from_file(path)
}
