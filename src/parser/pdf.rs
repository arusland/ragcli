use std::path::Path;

use anyhow::{Context, Result};

use super::DocumentParser;

pub struct PdfParser;

impl DocumentParser for PdfParser {
    fn supports(&self, path: &Path) -> bool {
        path.extension()
            .and_then(|e| e.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("pdf"))
    }

    fn parse(&self, path: &Path) -> Result<String> {
        pdf_extract::extract_text(path)
            .with_context(|| format!("failed to extract text from PDF {}", path.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supports_only_pdf_extension() {
        assert!(PdfParser.supports(Path::new("report.pdf")));
        assert!(PdfParser.supports(Path::new("REPORT.PDF")));
        assert!(!PdfParser.supports(Path::new("notes.txt")));
        assert!(!PdfParser.supports(Path::new("LICENSE")));
    }

    /// Builds a minimal single-page PDF containing `text`, computing the xref
    /// offsets so the file is well-formed.
    fn minimal_pdf(text: &str) -> Vec<u8> {
        let stream = format!("BT /F1 12 Tf 72 720 Td ({text}) Tj ET");
        let objects = [
            "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_string(),
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
             /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
                .to_string(),
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_string(),
            format!(
                "<< /Length {} >>\nstream\n{stream}\nendstream",
                stream.len()
            ),
        ];

        let mut pdf = String::from("%PDF-1.4\n");
        let mut offsets = Vec::new();
        for (i, body) in objects.iter().enumerate() {
            offsets.push(pdf.len());
            pdf.push_str(&format!("{} 0 obj\n{body}\nendobj\n", i + 1));
        }
        let xref_offset = pdf.len();
        pdf.push_str(&format!("xref\n0 {}\n", objects.len() + 1));
        pdf.push_str("0000000000 65535 f \n");
        for offset in offsets {
            pdf.push_str(&format!("{offset:010} 00000 n \n"));
        }
        pdf.push_str(&format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n",
            objects.len() + 1
        ));
        pdf.into_bytes()
    }

    #[test]
    fn parses_text_from_pdf() {
        let path = std::env::temp_dir().join(format!("ragcli-test-{}.pdf", std::process::id()));
        std::fs::write(&path, minimal_pdf("Hello PDF")).unwrap();
        let result = PdfParser.parse(&path);
        std::fs::remove_file(&path).ok();
        assert!(result.unwrap().contains("Hello PDF"));
    }

    #[test]
    fn errors_on_non_pdf_content() {
        let path =
            std::env::temp_dir().join(format!("ragcli-test-notpdf-{}.pdf", std::process::id()));
        std::fs::write(&path, b"just plain text, not a pdf").unwrap();
        let result = PdfParser.parse(&path);
        std::fs::remove_file(&path).ok();
        assert!(result.is_err());
    }
}
