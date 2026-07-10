use anyhow::{Context, Result};

pub const DEFAULT_EMBEDDING_MODEL: &str = "nomic-embed-text";
pub const DEFAULT_CHAT_MODEL: &str = "llama3.2";

pub struct Config {
    pub ollama_url: String,
    pub embedding_model: String,
    pub chat_model: String,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        let ollama_url = std::env::var("OLLAMA_URL").context(
            "OLLAMA_URL environment variable is not set. \
             Point it at your local Ollama server, e.g. http://localhost:11434",
        )?;
        Ok(Self {
            ollama_url: ollama_url.trim_end_matches('/').to_string(),
            embedding_model: embedding_model_from_env(),
            chat_model: chat_model_from_env(),
        })
    }
}

/// The configured embedding model, without requiring `OLLAMA_URL` to be set.
pub fn embedding_model_from_env() -> String {
    std::env::var("OLLAMA_EMBEDDING_MODEL").unwrap_or_else(|_| DEFAULT_EMBEDDING_MODEL.to_string())
}

/// The configured chat model, without requiring `OLLAMA_URL` to be set.
pub fn chat_model_from_env() -> String {
    std::env::var("OLLAMA_CHAT_MODEL").unwrap_or_else(|_| DEFAULT_CHAT_MODEL.to_string())
}
