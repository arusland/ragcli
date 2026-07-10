use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use super::EmbeddingProvider;

pub struct OllamaEmbedder {
    client: reqwest::blocking::Client,
    base_url: String,
    model: String,
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a [String],
}

#[derive(Deserialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

#[derive(Deserialize)]
struct ErrorResponse {
    error: String,
}

impl OllamaEmbedder {
    pub fn new(base_url: &str, model: &str) -> Self {
        Self {
            client: reqwest::blocking::Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
        }
    }
}

impl EmbeddingProvider for OllamaEmbedder {
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let url = format!("{}/api/embed", self.base_url);
        let response = self
            .client
            .post(&url)
            .json(&EmbedRequest {
                model: &self.model,
                input: texts,
            })
            .send()
            .with_context(|| format!("failed to reach Ollama at {url} (is the server running?)"))?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().unwrap_or_default();
            let detail = serde_json::from_str::<ErrorResponse>(&body)
                .map(|e| e.error)
                .unwrap_or(body);
            bail!(
                "Ollama returned {status} for model '{}': {detail}\n\
                 Hint: pull the model first with `ollama pull {}`",
                self.model,
                self.model
            );
        }

        let parsed: EmbedResponse = response
            .json()
            .context("failed to parse Ollama /api/embed response")?;
        if parsed.embeddings.len() != texts.len() {
            bail!(
                "Ollama returned {} embeddings for {} inputs",
                parsed.embeddings.len(),
                texts.len()
            );
        }
        Ok(parsed.embeddings)
    }
}
