use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use super::ChatProvider;

pub struct OllamaChat {
    client: reqwest::blocking::Client,
    base_url: String,
    model: String,
    verbose: bool,
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
            verbose: false,
        }
    }

    /// When enabled, prints the request and response bodies to stderr.
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
}

impl ChatProvider for OllamaChat {
    fn chat(&self, system: &str, user: &str) -> Result<String> {
        let url = format!("{}/api/chat", self.base_url);
        let request = ChatRequest {
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
        };
        if self.verbose {
            eprintln!(
                "POST {url}\n{}",
                serde_json::to_string_pretty(&request).unwrap_or_default()
            );
        }
        let response = self
            .client
            .post(&url)
            .json(&request)
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

        let body = response
            .text()
            .context("failed to read Ollama /api/chat response")?;
        if self.verbose {
            let pretty = serde_json::from_str::<serde_json::Value>(&body)
                .and_then(|value| serde_json::to_string_pretty(&value))
                .unwrap_or_else(|_| body.clone());
            eprintln!("Response from {url}:\n{pretty}");
        }
        let parsed: ChatResponse = serde_json::from_str(&body)
            .context("failed to parse Ollama /api/chat response")?;
        Ok(parsed.message.content)
    }
}
