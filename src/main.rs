mod chunker;
mod config;
mod embedding;
mod parser;
mod store;

use std::path::PathBuf;

use anyhow::{Result, bail};
use clap::{Parser, Subcommand};

use config::Config;
use embedding::{EmbeddingProvider, ollama::OllamaEmbedder};
use store::{EmbeddedChunk, VectorStore, sqlite::SqliteVectorStore};

const CHUNK_SIZE: usize = 1500;
const CHUNK_OVERLAP: usize = 200;

#[derive(Parser)]
#[command(name = "rag-cli", about = "Simple RAG CLI backed by SQLite and Ollama")]
struct Cli {
    /// Path to the vector database file
    #[arg(long, global = true, default_value = "rag.db")]
    db: PathBuf,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Parse a document, embed it, and store its vectors
    Add {
        /// Path to the document to add
        path: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Add { path } => add_document(&cli.db, &path),
    }
}

fn add_document(db_path: &PathBuf, doc_path: &PathBuf) -> Result<()> {
    let config = Config::from_env()?;

    let text = parser::parser_for(doc_path)?.parse(doc_path)?;
    let chunks = chunker::chunk_text(&text, CHUNK_SIZE, CHUNK_OVERLAP);
    if chunks.is_empty() {
        bail!("document {} contains no text", doc_path.display());
    }

    let embedder = OllamaEmbedder::new(&config.ollama_url, &config.embedding_model);
    let embeddings = embedder.embed(&chunks)?;
    let dim = embeddings.first().map_or(0, Vec::len);

    let embedded: Vec<EmbeddedChunk> = chunks
        .into_iter()
        .zip(embeddings)
        .enumerate()
        .map(|(index, (content, embedding))| EmbeddedChunk {
            index,
            content,
            embedding,
        })
        .collect();

    let mut store = SqliteVectorStore::open(db_path)?;
    store.init()?;
    store.add_document(&doc_path.to_string_lossy(), &embedded)?;

    println!(
        "Added {}: {} chunk(s), dim {}",
        doc_path.display(),
        embedded.len(),
        dim
    );
    Ok(())
}
