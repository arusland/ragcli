use std::path::Path;

use anyhow::{Context, Result};
use quick_xml::Reader;
use quick_xml::events::Event;

use super::DocumentParser;

pub struct XmlParser;

impl DocumentParser for XmlParser {
    fn supports(&self, path: &Path) -> bool {
        path.extension()
            .and_then(|e| e.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("xml"))
    }

    fn parse(&self, path: &Path) -> Result<String> {
        let xml = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read file {}", path.display()))?;
        extract_text(&xml)
    }
}

/// Strips markup from an XML document, keeping only its text content.
/// Attributes and tag names are dropped; each closed element ends its text
/// on its own line so sibling/nested content doesn't run together.
fn extract_text(xml: &str) -> Result<String> {
    let mut reader = Reader::from_str(xml);
    let mut text = String::new();
    loop {
        match reader.read_event().context("malformed XML")? {
            Event::Text(t) => {
                text.push_str(&t.xml10_content().context("malformed text in XML")?);
            }
            Event::CData(c) => {
                text.push_str(&String::from_utf8_lossy(&c.into_inner()));
            }
            Event::GeneralRef(r) => {
                if let Some(ch) = r
                    .resolve_char_ref()
                    .context("malformed character reference in XML")?
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
            Event::End(_) => text.push('\n'),
            Event::Eof => break,
            _ => {}
        }
    }
    let normalized = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    Ok(normalized)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supports_only_xml_extension() {
        assert!(XmlParser.supports(Path::new("data.xml")));
        assert!(XmlParser.supports(Path::new("DATA.XML")));
        assert!(!XmlParser.supports(Path::new("notes.txt")));
        assert!(!XmlParser.supports(Path::new("report.docx")));
    }

    fn write_temp(name: &str, contents: &str) -> std::path::PathBuf {
        let path =
            std::env::temp_dir().join(format!("ragcli-test-{}-{name}.xml", std::process::id()));
        std::fs::write(&path, contents).unwrap();
        path
    }

    #[test]
    fn strips_tags_from_nested_elements() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<root>
  <title>Hello World</title>
  <body>
    <p>First paragraph.</p>
    <p>Second paragraph.</p>
  </body>
</root>"#;
        let path = write_temp("nested", xml);
        let result = XmlParser.parse(&path);
        std::fs::remove_file(&path).ok();
        assert_eq!(
            result.unwrap(),
            "Hello World\nFirst paragraph.\nSecond paragraph."
        );
    }

    #[test]
    fn resolves_entity_and_character_references() {
        let xml = "<root><item>A&amp;B &lt;ok&gt; caf&#233;</item></root>";
        let path = write_temp("entities", xml);
        let result = XmlParser.parse(&path);
        std::fs::remove_file(&path).ok();
        assert_eq!(result.unwrap(), "A&B <ok> café");
    }

    #[test]
    fn ignores_attribute_values() {
        let xml = r#"<root><item id="secret" class="hidden">Visible</item></root>"#;
        let path = write_temp("attrs", xml);
        let result = XmlParser.parse(&path);
        std::fs::remove_file(&path).ok();
        assert_eq!(result.unwrap(), "Visible");
    }

    #[test]
    fn errors_on_malformed_xml() {
        let path = write_temp("malformed", "<root><unclosed></root>");
        let result = XmlParser.parse(&path);
        std::fs::remove_file(&path).ok();
        assert!(result.is_err());
    }
}
