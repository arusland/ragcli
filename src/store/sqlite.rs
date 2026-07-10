use std::path::Path;

use anyhow::{Context, Result, bail};
use rusqlite::{Connection, params};

use super::{EmbeddedChunk, SearchResult, StoredDocument, VectorStore, cosine_similarity};

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

/// Escapes LIKE wildcards so `term` matches literally under `ESCAPE '\'`.
fn escape_like(term: &str) -> String {
    term.replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_")
}

fn embedding_to_bytes(embedding: &[f32]) -> Vec<u8> {
    embedding.iter().flat_map(|v| v.to_le_bytes()).collect()
}

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

    fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<SearchResult>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT d.source_path, c.content, c.embedding, c.dim
                 FROM chunks c JOIN documents d ON d.id = c.document_id",
            )
            .context("failed to query chunks for similarity search")?;

        let mut results = Vec::new();
        let mut skipped = 0usize;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let dim: i64 = row.get(3)?;
            if dim as usize != query.len() {
                skipped += 1;
                continue;
            }
            let blob: Vec<u8> = row.get(2)?;
            results.push(SearchResult {
                source_path: row.get(0)?,
                content: row.get(1)?,
                score: cosine_similarity(query, &embedding_from_bytes(&blob)),
            });
        }

        if results.is_empty() && skipped > 0 {
            bail!(
                "no stored embeddings match the query dimension {} ({skipped} chunk(s) have a \
                 different dimension). Was the database built with a different embedding model? \
                 Re-add your documents with the current model.",
                query.len()
            );
        }

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(top_k);
        Ok(results)
    }

    fn document_count(&self) -> Result<usize> {
        let count: i64 = self
            .conn
            .query_row("SELECT count(*) FROM documents", [], |row| row.get(0))
            .context("failed to count documents")?;
        Ok(count as usize)
    }

    fn recent_documents(&self, limit: usize) -> Result<Vec<StoredDocument>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT d.source_path, datetime(d.added_at, 'localtime'), count(c.id)
                 FROM documents d LEFT JOIN chunks c ON c.document_id = d.id
                 GROUP BY d.id
                 ORDER BY d.added_at DESC, d.id DESC
                 LIMIT ?1",
            )
            .context("failed to query recent documents")?;

        let docs = stmt
            .query_map(params![limit as i64], |row| {
                Ok(StoredDocument {
                    source_path: row.get(0)?,
                    added_at: row.get(1)?,
                    chunk_count: row.get::<_, i64>(2)? as usize,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(docs)
    }

    fn find_documents(&self, term: &str) -> Result<Vec<StoredDocument>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT d.source_path, datetime(d.added_at, 'localtime'), count(c.id)
                 FROM documents d LEFT JOIN chunks c ON c.document_id = d.id
                 WHERE d.source_path LIKE '%' || ?1 || '%' ESCAPE '\\'
                 GROUP BY d.id
                 ORDER BY d.added_at DESC, d.id DESC",
            )
            .context("failed to query documents by substring")?;

        let docs = stmt
            .query_map(params![escape_like(term)], |row| {
                Ok(StoredDocument {
                    source_path: row.get(0)?,
                    added_at: row.get(1)?,
                    chunk_count: row.get::<_, i64>(2)? as usize,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(docs)
    }

    fn delete_document(&mut self, source_path: &str) -> Result<bool> {
        let deleted = self
            .conn
            .execute(
                "DELETE FROM documents WHERE source_path = ?1",
                params![source_path],
            )
            .with_context(|| format!("failed to delete document {source_path}"))?;
        Ok(deleted > 0)
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

    fn chunk(index: usize, content: &str, embedding: Vec<f32>) -> EmbeddedChunk {
        EmbeddedChunk {
            index,
            content: content.into(),
            embedding,
        }
    }

    #[test]
    fn search_ranks_by_similarity() {
        let mut store = SqliteVectorStore::open_in_memory().unwrap();
        store.init().unwrap();
        store
            .add_document("a.txt", &[chunk(0, "about apples", vec![1.0, 0.0, 0.0])])
            .unwrap();
        store
            .add_document("b.txt", &[chunk(0, "about bananas", vec![0.0, 1.0, 0.0])])
            .unwrap();

        let results = store.search(&[0.9, 0.1, 0.0], 5).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].source_path, "a.txt");
        assert_eq!(results[0].content, "about apples");
        assert!(results[0].score > 0.9);
        assert_eq!(results[1].source_path, "b.txt");
        assert!(results[1].score < results[0].score);
    }

    #[test]
    fn search_truncates_to_top_k() {
        let mut store = SqliteVectorStore::open_in_memory().unwrap();
        store.init().unwrap();
        store
            .add_document(
                "doc.txt",
                &[
                    chunk(0, "one", vec![1.0, 0.0]),
                    chunk(1, "two", vec![0.0, 1.0]),
                    chunk(2, "three", vec![1.0, 1.0]),
                ],
            )
            .unwrap();

        let results = store.search(&[1.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn search_on_empty_store_returns_empty() {
        let mut store = SqliteVectorStore::open_in_memory().unwrap();
        store.init().unwrap();
        assert!(store.search(&[1.0, 0.0], 5).unwrap().is_empty());
    }

    #[test]
    fn search_errors_when_all_dims_mismatch() {
        let mut store = SqliteVectorStore::open_in_memory().unwrap();
        store.init().unwrap();
        store.add_document("doc.txt", &sample_chunks()).unwrap();

        let err = store.search(&[1.0, 0.0], 5).unwrap_err();
        assert!(err.to_string().contains("dimension"));
    }

    #[test]
    fn document_count_reflects_stored_documents() {
        let mut store = SqliteVectorStore::open_in_memory().unwrap();
        store.init().unwrap();
        assert_eq!(store.document_count().unwrap(), 0);

        store.add_document("a.txt", &sample_chunks()).unwrap();
        store.add_document("b.txt", &sample_chunks()).unwrap();
        // re-adding must not double-count
        store.add_document("a.txt", &sample_chunks()).unwrap();
        assert_eq!(store.document_count().unwrap(), 2);
    }

    #[test]
    fn recent_documents_orders_newest_first_and_limits() {
        let mut store = SqliteVectorStore::open_in_memory().unwrap();
        store.init().unwrap();
        for name in ["a.txt", "b.txt", "c.txt"] {
            store.add_document(name, &sample_chunks()).unwrap();
        }
        // in-memory inserts land in the same second, so set distinct timestamps
        for (name, ts) in [
            ("a.txt", "2026-01-01 10:00:00"),
            ("b.txt", "2026-01-03 10:00:00"),
            ("c.txt", "2026-01-02 10:00:00"),
        ] {
            store
                .conn
                .execute(
                    "UPDATE documents SET added_at = ?1 WHERE source_path = ?2",
                    params![ts, name],
                )
                .unwrap();
        }

        let recent = store.recent_documents(2).unwrap();
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].source_path, "b.txt");
        // added_at is reported in local time, whatever zone the test runs in
        let expected: String = store
            .conn
            .query_row(
                "SELECT datetime('2026-01-03 10:00:00', 'localtime')",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(recent[0].added_at, expected);
        assert_eq!(recent[0].chunk_count, 2);
        assert_eq!(recent[1].source_path, "c.txt");
    }

    #[test]
    fn find_documents_matches_substring_newest_first() {
        let mut store = SqliteVectorStore::open_in_memory().unwrap();
        store.init().unwrap();
        for name in ["notes/a.txt", "notes/b.txt", "other.md"] {
            store.add_document(name, &sample_chunks()).unwrap();
        }
        // in-memory inserts land in the same second, so set distinct timestamps
        for (name, ts) in [
            ("notes/a.txt", "2026-01-01 10:00:00"),
            ("notes/b.txt", "2026-01-02 10:00:00"),
        ] {
            store
                .conn
                .execute(
                    "UPDATE documents SET added_at = ?1 WHERE source_path = ?2",
                    params![ts, name],
                )
                .unwrap();
        }

        let docs = store.find_documents("notes/").unwrap();
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0].source_path, "notes/b.txt");
        assert_eq!(docs[0].chunk_count, 2);
        assert_eq!(docs[1].source_path, "notes/a.txt");

        assert!(store.find_documents("missing").unwrap().is_empty());
        // the empty term matches every document ('*' at the CLI maps to it)
        assert_eq!(store.find_documents("").unwrap().len(), 3);
    }

    #[test]
    fn find_documents_treats_like_wildcards_literally() {
        let mut store = SqliteVectorStore::open_in_memory().unwrap();
        store.init().unwrap();
        store.add_document("100x.txt", &sample_chunks()).unwrap();
        store.add_document("100%.txt", &sample_chunks()).unwrap();

        let docs = store.find_documents("100%").unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].source_path, "100%.txt");

        let docs = store.find_documents("100_").unwrap();
        assert!(docs.is_empty());
    }

    #[test]
    fn delete_document_removes_document_and_chunks() {
        let mut store = SqliteVectorStore::open_in_memory().unwrap();
        store.init().unwrap();
        store.add_document("a.txt", &sample_chunks()).unwrap();
        store.add_document("b.txt", &sample_chunks()).unwrap();

        assert!(store.delete_document("a.txt").unwrap());
        assert_eq!(store.document_count().unwrap(), 1);
        let chunks: i64 = store
            .conn
            .query_row("SELECT count(*) FROM chunks", [], |row| row.get(0))
            .unwrap();
        assert_eq!(chunks, 2);

        assert!(!store.delete_document("unknown.txt").unwrap());
        assert_eq!(store.document_count().unwrap(), 1);
    }

    #[test]
    fn search_skips_mismatched_dims_but_returns_matching() {
        let mut store = SqliteVectorStore::open_in_memory().unwrap();
        store.init().unwrap();
        store
            .add_document("old.txt", &[chunk(0, "old model", vec![1.0, 0.0])])
            .unwrap();
        store
            .add_document("new.txt", &[chunk(0, "new model", vec![1.0, 0.0, 0.0])])
            .unwrap();

        let results = store.search(&[1.0, 0.0, 0.0], 5).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].source_path, "new.txt");
    }
}
