pub mod ollama;

use anyhow::Result;

/// Produces vector embeddings for texts. Implementations wrap a concrete
/// backend (Ollama by default).
pub trait EmbeddingProvider {
    /// Embeds each text; the result has one vector per input, in order.
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}
