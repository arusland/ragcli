# rag-cli

A simple RAG (Retrieval-Augmented Generation) CLI tool in Rust. Documents are parsed, split into chunks, embedded with a local [Ollama](https://ollama.com) model, and stored as vectors in SQLite.

## Requirements

- Rust toolchain (edition 2024)
- A running Ollama server with an embedding model and a chat model pulled:

  ```sh
  ollama pull nomic-embed-text
  ollama pull llama3.2
  ```

## Configuration

| Env var | Required | Default | Description |
|---|---|---|---|
| `OLLAMA_URL` | **yes** | — | Base URL of the Ollama server, e.g. `http://localhost:11434` |
| `OLLAMA_EMBEDDING_MODEL` | no | `nomic-embed-text` | Embedding model name |
| `OLLAMA_CHAT_MODEL` | no | `llama3.2` | Chat model used by `ask` |

## Usage

```powershell
$env:OLLAMA_URL = "http://localhost:11434"

# Parse a document, embed it, and store its vectors
cargo run -- add .\notes.txt
# Added .\notes.txt: 3 chunk(s), dim 768

# Answer a question using the stored documents as context
cargo run -- ask "What do my notes say about brewing?"
# <answer from the chat model>
#
# Sources:
# - .\notes.txt
```

Options:

- `--db <path>` — path to the SQLite database file (default: `./rag.db`)
- `--top-k <n>` — (`ask` only) number of most similar chunks to retrieve as context (default: 5)

Re-running `add` on the same file replaces its previously stored chunks, so the command is idempotent.

### Supported document types

| Parser | Extensions |
|---|---|
| `PlainTextParser` | `.txt`, `.md`, `.markdown`, `.log`, `.text`, and files without an extension |

Other file types are rejected with `unsupported document type`.

## How it works

```
add <path>
  └─ parser_for(path)          pick a DocumentParser by file type
  └─ chunk_text(...)           ~1500-char chunks, 200-char overlap,
                               split preferably at paragraph boundaries
  └─ EmbeddingProvider::embed  one batched POST {OLLAMA_URL}/api/embed
  └─ VectorStore::add_document upsert document + chunks in one transaction

ask <question>
  └─ EmbeddingProvider::embed  embed the question
  └─ VectorStore::search       cosine similarity over all chunks in Rust,
                               top-k best matches
  └─ ChatProvider::chat        question + numbered context chunks in one
                               non-streaming POST {OLLAMA_URL}/api/chat
```

Embeddings are stored in SQLite as little-endian `f32` BLOBs:

```sql
documents(id, source_path UNIQUE, added_at)
chunks(id, document_id → documents, chunk_index, content, embedding BLOB, dim)
```

## Extending

- **Another storage backend** (Qdrant, pgvector, ...): implement the `VectorStore` trait (`src/store/mod.rs`) and swap the instantiation in `main.rs`.
- **Another document type** (PDF, HTML, ...): implement the `DocumentParser` trait and register it in the `PARSERS` list (`src/parser/mod.rs`).
- **Another embedding backend**: implement the `EmbeddingProvider` trait (`src/embedding/mod.rs`).
- **Another chat backend**: implement the `ChatProvider` trait (`src/llm/mod.rs`).

## Development

```sh
cargo build
cargo test
```
