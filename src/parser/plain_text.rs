use std::path::Path;

use anyhow::{Context, Result};

use super::DocumentParser;

const TEXT_EXTENSIONS: &[&str] = &["txt", "md", "markdown", "log", "text"];

pub struct PlainTextParser;

impl DocumentParser for PlainTextParser {
    fn supports(&self, path: &Path) -> bool {
        match path.extension().and_then(|e| e.to_str()) {
            Some(ext) => TEXT_EXTENSIONS.contains(&ext.to_ascii_lowercase().as_str()),
            // Extensionless files are treated as plain text.
            None => true,
        }
    }

    fn parse(&self, path: &Path) -> Result<String> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("failed to read file {}", path.display()))?;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supports_text_files_and_extensionless() {
        assert!(PlainTextParser.supports(Path::new("a.txt")));
        assert!(PlainTextParser.supports(Path::new("a.MD")));
        assert!(PlainTextParser.supports(Path::new("LICENSE")));
        assert!(!PlainTextParser.supports(Path::new("a.pdf")));
    }
}
