use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use super::ChatProvider;

pub struct OllamaChat {
    client: reqwest::blocking::Client,
    base_url: String,
    model: String,
}

#[derive(Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    stream: bool,
}

#[derive(Deserialize)]
struct ChatResponse {
    message: ChatResponseMessage,
}

#[derive(Deserialize)]
struct ChatResponseMessage {
    content: String,
}

#[derive(Deserialize)]
struct ErrorResponse {
    error: String,
}

impl OllamaChat {
    pub fn new(base_url: &str, model: &str) -> Self {
        Self {
            client: reqwest::blocking::Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
        }
    }
}

impl ChatProvider for OllamaChat {
    fn chat(&self, system: &str, user: &str) -> Result<String> {
        let url = format!("{}/api/chat", self.base_url);
        let response = self
            .client
            .post(&url)
            .json(&ChatRequest {
                model: &self.model,
                messages: vec![
                    ChatMessage {
                        role: "system",
                        content: system,
                    },
                    ChatMessage {
                        role: "user",
                        content: user,
                    },
                ],
                // Ollama streams NDJSON by default; ask for a single JSON reply.
                stream: false,
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

        let parsed: ChatResponse = response
            .json()
            .context("failed to parse Ollama /api/chat response")?;
        Ok(parsed.message.content)
    }
}
