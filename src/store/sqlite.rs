use std::path::Path;

use anyhow::{Context, Result};
use rusqlite::{Connection, params};

use super::{EmbeddedChunk, VectorStore};

pub struct SqliteVectorStore {
    conn: Connection,
}

impl SqliteVectorStore {
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path)
            .with_context(|| format!("failed to open database at {}", path.display()))?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        Ok(Self { conn })
    }

    #[cfg(test)]
    fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        Ok(Self { conn })
    }
}

fn embedding_to_bytes(embedding: &[f32]) -> Vec<u8> {
    embedding.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[cfg(test)]
fn embedding_from_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

impl VectorStore for SqliteVectorStore {
    fn init(&mut self) -> Result<()> {
        self.conn
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS documents (
                     id INTEGER PRIMARY KEY,
                     source_path TEXT NOT NULL UNIQUE,
                     added_at TEXT NOT NULL DEFAULT (datetime('now'))
                 );
                 CREATE TABLE IF NOT EXISTS chunks (
                     id INTEGER PRIMARY KEY,
                     document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                     chunk_index INTEGER NOT NULL,
                     content TEXT NOT NULL,
                     embedding BLOB NOT NULL,
                     dim INTEGER NOT NULL
                 );",
            )
            .context("failed to create database schema")
    }

    fn add_document(&mut self, source_path: &str, chunks: &[EmbeddedChunk]) -> Result<()> {
        let tx = self.conn.transaction()?;

        tx.execute(
            "INSERT INTO documents (source_path) VALUES (?1)
             ON CONFLICT(source_path) DO UPDATE SET added_at = datetime('now')",
            params![source_path],
        )?;
        let document_id: i64 = tx.query_row(
            "SELECT id FROM documents WHERE source_path = ?1",
            params![source_path],
            |row| row.get(0),
        )?;

        tx.execute(
            "DELETE FROM chunks WHERE document_id = ?1",
            params![document_id],
        )?;

        let mut insert = tx.prepare(
            "INSERT INTO chunks (document_id, chunk_index, content, embedding, dim)
             VALUES (?1, ?2, ?3, ?4, ?5)",
        )?;
        for chunk in chunks {
            insert.execute(params![
                document_id,
                chunk.index as i64,
                chunk.content,
                embedding_to_bytes(&chunk.embedding),
                chunk.embedding.len() as i64,
            ])?;
        }
        drop(insert);

        tx.commit().context("failed to commit document")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_chunks() -> Vec<EmbeddedChunk> {
        vec![
            EmbeddedChunk {
                index: 0,
                content: "first chunk".into(),
                embedding: vec![0.1, 0.2, 0.3],
            },
            EmbeddedChunk {
                index: 1,
                content: "second chunk".into(),
                embedding: vec![-1.0, 0.5, 2.5],
            },
        ]
    }

    #[test]
    fn add_document_round_trips_embeddings() {
        let mut store = SqliteVectorStore::open_in_memory().unwrap();
        store.init().unwrap();
        store.add_document("doc.txt", &sample_chunks()).unwrap();

        let (content, blob, dim): (String, Vec<u8>, i64) = store
            .conn
            .query_row(
                "SELECT content, embedding, dim FROM chunks WHERE chunk_index = 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(content, "second chunk");
        assert_eq!(dim, 3);
        assert_eq!(embedding_from_bytes(&blob), vec![-1.0, 0.5, 2.5]);
    }

    #[test]
    fn readding_replaces_chunks() {
        let mut store = SqliteVectorStore::open_in_memory().unwrap();
        store.init().unwrap();
        store.add_document("doc.txt", &sample_chunks()).unwrap();
        store
            .add_document(
                "doc.txt",
                &[EmbeddedChunk {
                    index: 0,
                    content: "only chunk now".into(),
                    embedding: vec![1.0],
                }],
            )
            .unwrap();

        let docs: i64 = store
            .conn
            .query_row("SELECT count(*) FROM documents", [], |row| row.get(0))
            .unwrap();
        let chunks: i64 = store
            .conn
            .query_row("SELECT count(*) FROM chunks", [], |row| row.get(0))
            .unwrap();
        assert_eq!(docs, 1);
        assert_eq!(chunks, 1);
    }
}
