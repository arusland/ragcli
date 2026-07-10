pub mod sqlite;

use anyhow::Result;

/// A chunk of a document together with its embedding, ready to be stored.
pub struct EmbeddedChunk {
    pub index: usize,
    pub content: String,
    pub embedding: Vec<f32>,
}

/// A stored chunk returned from a similarity search, best match first.
#[derive(Debug)]
pub struct SearchResult {
    pub source_path: String,
    pub content: String,
    /// Cosine similarity to the query, in [-1, 1].
    pub score: f32,
}

/// Summary of a stored document, as reported by `recent_documents`.
#[derive(Debug)]
pub struct StoredDocument {
    pub source_path: String,
    pub chunk_count: usize,
    /// When the document was (last) added, in the machine's local time zone.
    pub added_at: String,
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

    /// Returns up to `top_k` stored chunks most similar to `query` by cosine
    /// similarity, best match first. Empty if the store has no chunks.
    fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<SearchResult>>;

    /// Returns the number of stored documents.
    fn document_count(&self) -> Result<usize>;

    /// Returns up to `limit` documents, most recently added first.
    fn recent_documents(&self, limit: usize) -> Result<Vec<StoredDocument>>;
}

pub(crate) fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical_vectors_is_one() {
        let v = [1.0, 2.0, 3.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal_vectors_is_zero() {
        assert_eq!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]), 0.0);
    }

    #[test]
    fn cosine_opposite_vectors_is_negative_one() {
        assert!((cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_zero_vector_is_zero() {
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
    }
}
