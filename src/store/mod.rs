pub mod sqlite;

use anyhow::Result;

/// A chunk of a document together with its embedding, ready to be stored.
pub struct EmbeddedChunk {
    pub index: usize,
    pub content: String,
    pub embedding: Vec<f32>,
}

/// Storage backend for document embeddings. The default implementation is
/// SQLite; alternative backends (Qdrant, pgvector, ...) only need to
/// implement this trait.
pub trait VectorStore {
    /// Creates the schema / collection if it does not exist yet.
    fn init(&mut self) -> Result<()>;

    /// Upserts a document and its chunk embeddings. Chunks previously stored
    /// for the same `source_path` are replaced.
    fn add_document(&mut self, source_path: &str, chunks: &[EmbeddedChunk]) -> Result<()>;
}
