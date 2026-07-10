mod chunker;
mod config;
mod embedding;
mod llm;
mod parser;
mod store;

use std::path::PathBuf;

use anyhow::{Result, anyhow, bail};
use clap::{Parser, Subcommand};

use config::Config;
use embedding::{EmbeddingProvider, ollama::OllamaEmbedder};
use llm::{ChatProvider, ollama::OllamaChat};
use store::{EmbeddedChunk, SearchResult, VectorStore, sqlite::SqliteVectorStore};

const CHUNK_SIZE: usize = 1500;
const CHUNK_OVERLAP: usize = 200;

const SYSTEM_PROMPT: &str = "You are a helpful assistant. Answer the user's question using ONLY \
    the provided context. If the context does not contain the answer, say you don't know rather \
    than guessing. Be concise.";

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
    /// Answer a question using the stored documents as context
    Ask {
        /// The question to answer
        question: String,
        /// Number of most similar chunks to retrieve as context
        #[arg(long, default_value_t = 5)]
        top_k: usize,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Add { path } => add_document(&cli.db, &path),
        Command::Ask { question, top_k } => ask_question(&cli.db, &question, top_k),
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

fn ask_question(db_path: &PathBuf, question: &str, top_k: usize) -> Result<()> {
    let config = Config::from_env()?;

    let embedder = OllamaEmbedder::new(&config.ollama_url, &config.embedding_model);
    let query = embedder
        .embed(&[question.to_string()])?
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("Ollama returned no embedding for the question"))?;

    let mut store = SqliteVectorStore::open(db_path)?;
    store.init()?;
    let results = store.search(&query, top_k)?;
    if results.is_empty() {
        bail!(
            "no documents found in {} — add one first with `rag-cli add <path>`",
            db_path.display()
        );
    }

    let chat = OllamaChat::new(&config.ollama_url, &config.chat_model);
    let answer = chat.chat(SYSTEM_PROMPT, &build_prompt(question, &results))?;

    println!("{answer}");
    println!("\nSources:");
    let mut seen = std::collections::HashSet::new();
    for result in &results {
        if seen.insert(result.source_path.as_str()) {
            println!("- {}", result.source_path);
        }
    }
    Ok(())
}

fn build_prompt(question: &str, results: &[SearchResult]) -> String {
    let mut prompt = String::from("Context:\n");
    for (i, result) in results.iter().enumerate() {
        prompt.push_str(&format!(
            "\n[{}] ({})\n{}\n",
            i + 1,
            result.source_path,
            result.content
        ));
    }
    prompt.push_str(&format!("\nQuestion: {question}"));
    prompt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_prompt_numbers_chunks_and_includes_question() {
        let results = vec![
            SearchResult {
                source_path: "a.txt".into(),
                content: "apples are red".into(),
                score: 0.9,
            },
            SearchResult {
                source_path: "b.txt".into(),
                content: "bananas are yellow".into(),
                score: 0.5,
            },
        ];
        let prompt = build_prompt("What color are apples?", &results);
        assert!(prompt.contains("[1] (a.txt)\napples are red"));
        assert!(prompt.contains("[2] (b.txt)\nbananas are yellow"));
        assert!(prompt.ends_with("Question: What color are apples?"));
    }
}
