# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`ragcli` is a Rust CLI RAG tool: it parses documents, chunks the text, embeds chunks via a local Ollama server, and stores vectors in SQLite. See README.md for usage.

## Commands

```sh
cargo build
cargo test                          # all unit tests (colocated in modules)
cargo test chunker::                # tests of a single module
cargo test readding_replaces       # a single test by name substring
```

Running the binary requires `OLLAMA_URL` (mandatory; e.g. `http://localhost:11434`) and optionally `OLLAMA_EMBEDDING_MODEL` (default `nomic-embed-text`) and `OLLAMA_CHAT_MODEL` (default `llama3.2`):

```sh
OLLAMA_URL=http://localhost:11434 cargo run -- add path/to/doc.txt [--db rag.db]
OLLAMA_URL=http://localhost:11434 cargo run -- ask "question" [--top-k 5] [--db rag.db] [--verbose]
cargo run -- status [--db rag.db]   # needs no OLLAMA_URL
```

Tests need no Ollama server. For manual end-to-end testing without a real model, a stub HTTP server answering `POST /api/embed` with `{"embeddings": [[...], ...]}` and `POST /api/chat` with `{"message": {"content": "..."}}` is sufficient.

## Architecture

Everything is synchronous by design (`reqwest::blocking`, `rusqlite`); do not introduce async without cause.

The codebase is built around four traits, each in its module's `mod.rs`, with implementations in sibling files. `main.rs` is the only wiring point — it instantiates concrete types and passes them around as traits:

- **`VectorStore`** (`src/store/mod.rs`) — storage abstraction (`init`, `add_document`, `search`, plus `document_count`/`recent_documents` used by the `status` command). `SqliteVectorStore` (`src/store/sqlite.rs`) is the default: embeddings are little-endian `f32` BLOBs in a `documents`/`chunks` schema; `add_document` is an upsert that deletes prior chunks for the same `source_path` inside one transaction; `search` is a brute-force cosine-similarity scan over all chunks in Rust (chunks whose `dim` doesn't match the query are skipped; it errors only if all chunks mismatch). A new backend (Qdrant, pgvector, ...) implements this trait and gets wired in `main.rs`.
- **`DocumentParser`** (`src/parser/mod.rs`) — one implementation per document type. `parser_for(path)` walks the static `PARSERS` registry and returns the first parser whose `supports(path)` matches; a new parser must be added to that list. `PlainTextParser` claims text extensions *and* extensionless files, so order matters if a later parser also wants extensionless files.
- **`EmbeddingProvider`** (`src/embedding/mod.rs`) — `embed(&[String]) -> Vec<Vec<f32>>`, one vector per input, in order. `OllamaEmbedder` sends a single batched request to `{OLLAMA_URL}/api/embed`.
- **`ChatProvider`** (`src/llm/mod.rs`) — `chat(system, user) -> String`. `OllamaChat` (`src/llm/ollama.rs`) sends one non-streaming (`"stream": false`) request to `{OLLAMA_URL}/api/chat`; the RAG prompt itself is built in `main.rs` (`build_prompt`).

`src/chunker.rs` is a free function, not a trait: ~1500-char chunks with 200-char overlap (constants in `main.rs`), preferring paragraph > newline > whitespace breaks within the last third of the window. Chunks are trimmed; because of the overlap, consecutive chunks intentionally share content — keep this in mind when asserting on chunk boundaries in tests.

## Workflow

- After each change, generate a short one-line commit message describing it (e.g. `fix: handle empty chunks`). Do not commit unless explicitly asked.

## Conventions

- Errors: `anyhow` everywhere, with `.context()`/`bail!` messages that tell the user what to do (e.g. which env var to set, `ollama pull` hint).
- Unit tests are colocated `#[cfg(test)] mod tests` blocks; SQLite tests use `open_in_memory()`.
