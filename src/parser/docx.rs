use std::io::Read;
use std::path::Path;

use anyhow::{Context, Result};
use quick_xml::Reader;
use quick_xml::events::Event;

use super::DocumentParser;

pub struct DocxParser;

impl DocumentParser for DocxParser {
    fn supports(&self, path: &Path) -> bool {
        path.extension()
            .and_then(|e| e.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("docx"))
    }

    fn parse(&self, path: &Path) -> Result<String> {
        let file = std::fs::File::open(path)
            .with_context(|| format!("failed to open file {}", path.display()))?;
        let mut archive = zip::ZipArchive::new(file)
            .with_context(|| format!("{} is not a valid .docx (zip) file", path.display()))?;
        let mut xml = String::new();
        archive
            .by_name("word/document.xml")
            .with_context(|| {
                format!(
                    "{} is not a Word document (missing word/document.xml)",
                    path.display()
                )
            })?
            .read_to_string(&mut xml)
            .with_context(|| format!("failed to read document body of {}", path.display()))?;
        extract_text(&xml)
    }
}

/// Pulls the visible text out of a WordprocessingML body: the content of
/// `w:t` runs, with tabs/line breaks for `w:tab`/`w:br` and a blank line
/// after each paragraph (`w:p`) so the chunker sees paragraph boundaries.
fn extract_text(xml: &str) -> Result<String> {
    let mut reader = Reader::from_str(xml);
    let mut text = String::new();
    let mut in_run_text = false;
    loop {
        match reader.read_event().context("malformed document.xml")? {
            Event::Start(e) if e.local_name().as_ref() == b"t" => in_run_text = true,
            Event::End(e) => match e.local_name().as_ref() {
                b"t" => in_run_text = false,
                b"p" => text.push_str("\n\n"),
                _ => {}
            },
            Event::Empty(e) => match e.local_name().as_ref() {
                b"tab" => text.push('\t'),
                b"br" => text.push('\n'),
                _ => {}
            },
            Event::Text(t) if in_run_text => {
                text.push_str(
                    &t.xml10_content()
                        .context("malformed text in document.xml")?,
                );
            }
            // Entity references arrive as separate events, not as text.
            Event::GeneralRef(r) if in_run_text => {
                if let Some(ch) = r
                    .resolve_char_ref()
                    .context("malformed character reference in document.xml")?
                {
                    text.push(ch);
                } else {
                    match r.xml10_content()?.as_ref() {
                        "amp" => text.push('&'),
                        "lt" => text.push('<'),
                        "gt" => text.push('>'),
                        "quot" => text.push('"'),
                        "apos" => text.push('\''),
                        _ => {} // custom entities: nothing to resolve them with
                    }
                }
            }
            Event::Eof => break,
            _ => {}
        }
    }
    Ok(text.trim().to_string())
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    #[test]
    fn supports_only_docx_extension() {
        assert!(DocxParser.supports(Path::new("report.docx")));
        assert!(DocxParser.supports(Path::new("REPORT.DOCX")));
        assert!(!DocxParser.supports(Path::new("legacy.doc")));
        assert!(!DocxParser.supports(Path::new("notes.txt")));
        assert!(!DocxParser.supports(Path::new("LICENSE")));
    }

    fn minimal_docx(body_xml: &str) -> Vec<u8> {
        let document = format!(
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:body>{body_xml}</w:body>
</w:document>"#
        );
        let mut zip = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
        zip.start_file(
            "word/document.xml",
            zip::write::SimpleFileOptions::default(),
        )
        .unwrap();
        zip.write_all(document.as_bytes()).unwrap();
        zip.finish().unwrap().into_inner()
    }

    fn write_temp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let path =
            std::env::temp_dir().join(format!("ragcli-test-{}-{name}.docx", std::process::id()));
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn parses_paragraphs_tabs_and_breaks() {
        let docx = minimal_docx(
            "<w:p><w:r><w:t>Hello</w:t></w:r><w:r><w:t xml:space=\"preserve\"> Word</w:t></w:r></w:p>\
             <w:p><w:r><w:t>Col1</w:t><w:tab/><w:t>Col2</w:t><w:br/><w:t>Line2</w:t></w:r></w:p>",
        );
        let path = write_temp("body", &docx);
        let result = DocxParser.parse(&path);
        std::fs::remove_file(&path).ok();
        assert_eq!(result.unwrap(), "Hello Word\n\nCol1\tCol2\nLine2");
    }

    #[test]
    fn resolves_entity_and_character_references() {
        let docx = minimal_docx("<w:p><w:r><w:t>A&amp;B &lt;ok&gt; caf&#233;</w:t></w:r></w:p>");
        let path = write_temp("entities", &docx);
        let result = DocxParser.parse(&path);
        std::fs::remove_file(&path).ok();
        assert_eq!(result.unwrap(), "A&B <ok> café");
    }

    #[test]
    fn ignores_text_outside_runs() {
        let docx = minimal_docx(
            "<w:p><w:pPr><w:instrText>PAGEREF _Toc1</w:instrText></w:pPr>\
             <w:r><w:t>Visible</w:t></w:r></w:p>",
        );
        let path = write_temp("outside", &docx);
        let result = DocxParser.parse(&path);
        std::fs::remove_file(&path).ok();
        assert_eq!(result.unwrap(), "Visible");
    }

    #[test]
    fn errors_on_non_docx_content() {
        let path = write_temp("notzip", b"just plain text, not a zip");
        let result = DocxParser.parse(&path);
        std::fs::remove_file(&path).ok();
        assert!(result.is_err());
    }

    #[test]
    fn errors_on_zip_without_document_xml() {
        let mut zip = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
        zip.start_file("other.txt", zip::write::SimpleFileOptions::default())
            .unwrap();
        zip.write_all(b"nope").unwrap();
        let bytes = zip.finish().unwrap().into_inner();
        let path = write_temp("nodoc", &bytes);
        let result = DocxParser.parse(&path);
        std::fs::remove_file(&path).ok();
        assert!(result.unwrap_err().to_string().contains("Word document"));
    }
}
