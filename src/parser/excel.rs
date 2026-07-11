use std::path::Path;

use anyhow::{Context, Result};
use calamine::{Data, Reader, open_workbook_auto};

use super::DocumentParser;

const EXCEL_EXTENSIONS: &[&str] = &["xlsx", "xlsm", "xlsb", "xls"];

pub struct ExcelParser;

impl DocumentParser for ExcelParser {
    fn supports(&self, path: &Path) -> bool {
        path.extension()
            .and_then(|e| e.to_str())
            .is_some_and(|ext| EXCEL_EXTENSIONS.contains(&ext.to_ascii_lowercase().as_str()))
    }

    /// Renders each sheet as tab-separated rows under a `Sheet: <name>`
    /// heading, so cell adjacency survives into the embedded text.
    fn parse(&self, path: &Path) -> Result<String> {
        let mut workbook = open_workbook_auto(path)
            .with_context(|| format!("failed to open Excel file {}", path.display()))?;
        let mut text = String::new();
        for name in workbook.sheet_names() {
            let range = workbook
                .worksheet_range(&name)
                .with_context(|| format!("failed to read sheet '{name}' of {}", path.display()))?;
            let mut sheet = String::new();
            for row in range.rows() {
                let cells: Vec<String> = row
                    .iter()
                    .map(|cell| match cell {
                        Data::Empty => String::new(),
                        other => other.to_string(),
                    })
                    .collect();
                let line = cells.join("\t");
                if !line.trim().is_empty() {
                    sheet.push_str(line.trim_end());
                    sheet.push('\n');
                }
            }
            if !sheet.is_empty() {
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(&format!("Sheet: {name}\n{sheet}"));
            }
        }
        Ok(text.trim_end().to_string())
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    #[test]
    fn supports_excel_extensions() {
        assert!(ExcelParser.supports(Path::new("data.xlsx")));
        assert!(ExcelParser.supports(Path::new("DATA.XLSX")));
        assert!(ExcelParser.supports(Path::new("macro.xlsm")));
        assert!(ExcelParser.supports(Path::new("legacy.xls")));
        assert!(!ExcelParser.supports(Path::new("notes.txt")));
        assert!(!ExcelParser.supports(Path::new("LICENSE")));
    }

    /// Builds a minimal one-sheet xlsx using inline strings (no
    /// sharedStrings.xml needed).
    fn minimal_xlsx(sheet_data_xml: &str) -> Vec<u8> {
        let entries: &[(&str, String)] = &[
            (
                "[Content_Types].xml",
                r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>"#
                    .to_string(),
            ),
            (
                "_rels/.rels",
                r#"<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"#
                    .to_string(),
            ),
            (
                "xl/workbook.xml",
                r#"<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets>
</workbook>"#
                    .to_string(),
            ),
            (
                "xl/_rels/workbook.xml.rels",
                r#"<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>"#
                    .to_string(),
            ),
            (
                "xl/worksheets/sheet1.xml",
                format!(
                    r#"<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<sheetData>{sheet_data_xml}</sheetData>
</worksheet>"#
                ),
            ),
        ];
        let mut zip = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
        for (name, content) in entries {
            zip.start_file(*name, zip::write::SimpleFileOptions::default())
                .unwrap();
            zip.write_all(content.as_bytes()).unwrap();
        }
        zip.finish().unwrap().into_inner()
    }

    fn write_temp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let path =
            std::env::temp_dir().join(format!("ragcli-test-{}-{name}.xlsx", std::process::id()));
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn parses_rows_as_tab_separated_lines() {
        let xlsx = minimal_xlsx(
            r#"<row r="1"><c r="A1" t="inlineStr"><is><t>Name</t></is></c><c r="B1" t="inlineStr"><is><t>Age</t></is></c></row>
<row r="2"><c r="A2" t="inlineStr"><is><t>Alice</t></is></c><c r="B2"><v>30</v></c></row>"#,
        );
        let path = write_temp("rows", &xlsx);
        let result = ExcelParser.parse(&path);
        std::fs::remove_file(&path).ok();
        assert_eq!(result.unwrap(), "Sheet: Sheet1\nName\tAge\nAlice\t30");
    }

    #[test]
    fn errors_on_non_excel_content() {
        let path = write_temp("notxlsx", b"just plain text, not a workbook");
        let result = ExcelParser.parse(&path);
        std::fs::remove_file(&path).ok();
        assert!(result.is_err());
    }
}
