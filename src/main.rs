mod chunker;
mod config;
mod embedding;
mod llm;
mod parser;
mod store;

use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
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
#[command(name = "ragcli", about = "Simple RAG CLI backed by SQLite and Ollama")]
struct Cli {
    /// Path to the vector database file
    #[arg(long, global = true, default_value = "rag.db")]
    db: PathBuf,

    /// Print diagnostic details (retrieved chunks, LLM request/response) to stderr
    #[arg(long, short = 'v', global = true)]
    verbose: bool,

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
    /// Show the configured models and what the database contains
    Status,
    /// List stored documents whose path contains TERM, most recently updated first
    Doc {
        /// Substring to match against stored document paths; '*' lists all documents
        term: String,
        /// Delete the matched documents
        #[arg(long)]
        rm: bool,
        /// With --rm: delete without asking for confirmation
        #[arg(long, requires = "rm")]
        force: bool,
    },
}

const RECENT_DOCUMENTS: usize = 5;

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Add { path } => add_document(&cli.db, &path, cli.verbose),
        Command::Ask { question, top_k } => ask_question(&cli.db, &question, top_k, cli.verbose),
        Command::Status => show_status(&cli.db),
        Command::Doc { term, rm, force } => doc_command(&cli.db, &term, rm, force),
    }
}

fn doc_command(db_path: &Path, term: &str, rm: bool, force: bool) -> Result<()> {
    let mut store = SqliteVectorStore::open(db_path)?;
    store.init()?;

    // '*' lists everything; the empty substring matches every path
    let docs = store.find_documents(if term == "*" { "" } else { term })?;
    if docs.is_empty() {
        println!("no documents match '{term}'");
        return Ok(());
    }
    for doc in &docs {
        match &doc.error {
            Some(error) => println!(
                "{}  {} ({} chunk(s), error: {})",
                doc.added_at, doc.source_path, doc.chunk_count, error
            ),
            None => println!(
                "{}  {} ({} chunk(s))",
                doc.added_at, doc.source_path, doc.chunk_count
            ),
        }
    }

    if rm {
        if !force {
            print!("Delete {} document(s)? [y/N]: ", docs.len());
            std::io::Write::flush(&mut std::io::stdout())?;
            let mut answer = String::new();
            std::io::stdin().read_line(&mut answer)?;
            if !is_yes(&answer) {
                println!("aborted");
                return Ok(());
            }
        }
        for doc in &docs {
            store.delete_document(&doc.source_path)?;
        }
        println!("Deleted {} document(s)", docs.len());
    }
    Ok(())
}

fn is_yes(answer: &str) -> bool {
    matches!(answer.trim().to_lowercase().as_str(), "y" | "yes")
}

fn show_status(db_path: &Path) -> Result<()> {
    let mut store = SqliteVectorStore::open(db_path)?;
    store.init()?;

    println!("Database:        {}", db_path.display());
    println!("Embedding model: {}", config::embedding_model_from_env());
    println!("Chat model:      {}", config::chat_model_from_env());
    println!("Documents:       {}", store.document_count()?);

    let recent = store.recent_documents(RECENT_DOCUMENTS)?;
    if !recent.is_empty() {
        println!("\nRecent documents:");
        for doc in recent {
            println!(
                "  {}  {} ({} chunk(s))",
                doc.added_at, doc.source_path, doc.chunk_count
            );
        }
    }
    Ok(())
}

/// Absolute, normalized form of `path`, used as the document's `source_path` key
/// so the same file is never stored under several spellings. Prefers
/// `canonicalize` (resolves symlinks); for paths that do not exist it falls back
/// to lexical normalization so read failures are still recorded under a stable key.
fn normalize_source_path(path: &Path) -> Result<PathBuf> {
    if let Ok(canonical) = path.canonicalize() {
        return Ok(canonical);
    }
    let absolute = std::path::absolute(path)
        .with_context(|| format!("failed to resolve {}", path.display()))?;
    let mut normalized = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            other => normalized.push(other),
        }
    }
    Ok(normalized)
}

fn add_document(db_path: &Path, doc_path: &Path, verbose: bool) -> Result<()> {
    let config = Config::from_env()?;

    let mut store = SqliteVectorStore::open(db_path)?;
    store.init()?;

    let doc_path = &normalize_source_path(doc_path)?;
    let source_path = doc_path.to_string_lossy();
    let bytes = match std::fs::read(doc_path)
        .with_context(|| format!("failed to read {}", doc_path.display()))
    {
        Ok(bytes) => bytes,
        Err(err) => {
            store.set_document_error(&source_path, &format!("{err:#}"))?;
            return Err(err);
        }
    };
    let size = bytes.len() as u64;
    let hash = format!("{:x}", md5::compute(&bytes));

    if store.document_fingerprint(&source_path)? == Some((size, hash.clone())) {
        println!(
            "Skipped {}: already added and unchanged (size {size} bytes, md5 {hash})",
            doc_path.display()
        );
        return Ok(());
    }

    let text = match parser::parser_for(doc_path).and_then(|p| p.parse(doc_path)) {
        Ok(text) => text,
        Err(err) => {
            store.set_document_error(&source_path, &format!("{err:#}"))?;
            return Err(err);
        }
    };
    let chunks = chunker::chunk_text(&text, CHUNK_SIZE, CHUNK_OVERLAP);
    if chunks.is_empty() {
        bail!("document {} contains no text", doc_path.display());
    }

    if verbose {
        eprintln!(
            "Embedding {} chunk(s) with model '{}' at {}",
            chunks.len(),
            config.embedding_model,
            config.ollama_url
        );
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

    store.add_document(&source_path, size, &hash, &embedded)?;

    println!(
        "Added {}: {} chunk(s), dim {}",
        doc_path.display(),
        embedded.len(),
        dim
    );
    Ok(())
}

fn ask_question(db_path: &Path, question: &str, top_k: usize, verbose: bool) -> Result<()> {
    let config = Config::from_env()?;

    if verbose {
        eprintln!(
            "Embedding question with model '{}' at {}",
            config.embedding_model, config.ollama_url
        );
    }
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
            "no documents found in {} — add one first with `ragcli add <path>`",
            db_path.display()
        );
    }

    if verbose {
        eprint!("{}", format_retrieved(&results, top_k));
    }

    let chat = OllamaChat::new(&config.ollama_url, &config.chat_model).with_verbose(verbose);
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

const PREVIEW_CHARS: usize = 200;

fn format_retrieved(results: &[SearchResult], top_k: usize) -> String {
    let mut out = format!("Retrieved {} chunk(s) (top-k = {top_k}):\n", results.len());
    for (i, result) in results.iter().enumerate() {
        let collapsed = result
            .content
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        let mut preview: String = collapsed.chars().take(PREVIEW_CHARS).collect();
        if collapsed.chars().count() > PREVIEW_CHARS {
            preview.push('…');
        }
        out.push_str(&format!(
            "[{}] score {:.4}  {}\n    {}\n",
            i + 1,
            result.score,
            result.source_path,
            preview
        ));
    }
    out
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
    fn normalize_source_path_makes_existing_relative_path_absolute_and_clean() {
        // tests run with the crate root as CWD, where Cargo.toml exists
        let normalized = normalize_source_path(Path::new("./Cargo.toml")).unwrap();
        assert!(normalized.is_absolute());
        assert_eq!(normalized.file_name().unwrap(), "Cargo.toml");
        assert!(
            normalized
                .components()
                .all(|c| !matches!(c, Component::CurDir | Component::ParentDir))
        );
    }

    #[test]
    fn normalize_source_path_lexically_resolves_missing_paths() {
        let normalized = normalize_source_path(Path::new("no-such-dir/../missing.txt")).unwrap();
        assert_eq!(
            normalized,
            std::env::current_dir().unwrap().join("missing.txt")
        );
    }

    #[test]
    fn is_yes_accepts_only_y_or_yes_case_insensitively() {
        for answer in ["y", "Y", "yes", "YES", " y \n"] {
            assert!(is_yes(answer), "{answer:?} should be yes");
        }
        for answer in ["", "\n", "n", "no", "yess", "sure"] {
            assert!(!is_yes(answer), "{answer:?} should not be yes");
        }
    }

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

    #[test]
    fn format_retrieved_numbers_chunks_with_scores_and_truncates_previews() {
        let results = vec![
            SearchResult {
                source_path: "a.txt".into(),
                content: "apples\nare  red".into(),
                score: 0.9,
            },
            SearchResult {
                source_path: "b.txt".into(),
                content: "x".repeat(PREVIEW_CHARS + 50),
                score: 0.5,
            },
        ];
        let out = format_retrieved(&results, 5);
        assert!(out.starts_with("Retrieved 2 chunk(s) (top-k = 5):\n"));
        // newlines and repeated spaces are collapsed in the preview
        assert!(out.contains("[1] score 0.9000  a.txt\n    apples are red\n"));
        assert!(out.contains("[2] score 0.5000  b.txt\n"));
        let long_preview = format!("{}…", "x".repeat(PREVIEW_CHARS));
        assert!(out.contains(&long_preview));
    }
}
