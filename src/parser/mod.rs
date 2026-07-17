pub mod code;
pub mod docx;
pub mod excel;
pub mod pdf;
pub mod plain_text;
pub mod xml;

use std::path::Path;

use anyhow::{Result, bail};

use code::CodeParser;
use docx::DocxParser;
use excel::ExcelParser;
use pdf::PdfParser;
use plain_text::PlainTextParser;
use xml::XmlParser;

/// Extracts plain text from a document file. One implementation per document
/// type (plain text, PDF, Word, Excel now; HTML, ... later).
pub trait DocumentParser {
    /// Returns true if this parser can handle the given file.
    fn supports(&self, path: &Path) -> bool;

    /// Extracts the document's text content.
    fn parse(&self, path: &Path) -> Result<String>;
}

// PlainTextParser claims extensionless files, so it must stay first if a
// later parser also wants them.
static PARSERS: &[&(dyn DocumentParser + Sync)] = &[
    &PlainTextParser,
    &PdfParser,
    &DocxParser,
    &ExcelParser,
    &XmlParser,
    &CodeParser,
];

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
    fn selects_pdf_parser_for_pdf_files() {
        assert!(parser_for(Path::new("report.pdf")).is_ok());
    }

    #[test]
    fn selects_office_parsers_for_word_and_excel_files() {
        assert!(parser_for(Path::new("report.docx")).is_ok());
        assert!(parser_for(Path::new("data.xlsx")).is_ok());
        assert!(parser_for(Path::new("legacy.xls")).is_ok());
    }

    #[test]
    fn selects_xml_parser_for_xml_files() {
        assert!(parser_for(Path::new("data.xml")).is_ok());
    }

    #[test]
    fn selects_code_parser_for_source_files() {
        assert!(parser_for(Path::new("Main.java")).is_ok());
        assert!(parser_for(Path::new("script.py")).is_ok());
        assert!(parser_for(Path::new("build.gradle")).is_ok());
        assert!(parser_for(Path::new("deploy.sh")).is_ok());
        assert!(parser_for(Path::new("run.bat")).is_ok());
    }

    #[test]
    fn rejects_unknown_extensions() {
        assert!(parser_for(Path::new("image.png")).is_err());
        // Legacy binary .doc is not supported (docx only).
        assert!(parser_for(Path::new("legacy.doc")).is_err());
    }
}
