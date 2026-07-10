pub mod plain_text;

use std::path::Path;

use anyhow::{Result, bail};

use plain_text::PlainTextParser;

/// Extracts plain text from a document file. One implementation per document
/// type (plain text now; PDF, HTML, ... later).
pub trait DocumentParser {
    /// Returns true if this parser can handle the given file.
    fn supports(&self, path: &Path) -> bool;

    /// Extracts the document's text content.
    fn parse(&self, path: &Path) -> Result<String>;
}

static PARSERS: &[&(dyn DocumentParser + Sync)] = &[&PlainTextParser];

/// Returns the first registered parser that supports `path`.
pub fn parser_for(path: &Path) -> Result<&'static dyn DocumentParser> {
    for parser in PARSERS {
        if parser.supports(path) {
            return Ok(*parser);
        }
    }
    bail!("unsupported document type: {}", path.display())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selects_plain_text_parser_for_text_extensions() {
        assert!(parser_for(Path::new("notes.txt")).is_ok());
        assert!(parser_for(Path::new("README.md")).is_ok());
    }

    #[test]
    fn rejects_unknown_extensions() {
        assert!(parser_for(Path::new("report.pdf")).is_err());
    }
}
