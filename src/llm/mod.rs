pub mod ollama;

use anyhow::Result;

/// Text-generation backend used to answer questions. The default
/// implementation is Ollama; alternative backends only need to implement
/// this trait.
pub trait ChatProvider {
    /// Generates an answer from a system prompt and a user prompt.
    fn chat(&self, system: &str, user: &str) -> Result<String>;
}
