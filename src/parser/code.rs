use std::path::Path;

use anyhow::Result;

use super::DocumentParser;
use super::plain_text::PlainTextParser;

const CODE_EXTENSIONS: &[&str] = &[
    // JVM
    "java",
    "kt",
    "kts",
    "groovy",
    "gradle",
    "scala",
    "clj",
    // Scripting
    "py",
    "rb",
    "pl",
    "php",
    "lua",
    "r",
    // Shell / Windows scripts
    "sh",
    "bash",
    "zsh",
    "fish",
    "ps1",
    "psm1",
    "bat",
    "cmd",
    // C family
    "c",
    "h",
    "cc",
    "cpp",
    "cxx",
    "hpp",
    "hh",
    "cs",
    "m",
    "mm",
    // Web
    "js",
    "jsx",
    "mjs",
    "cjs",
    "ts",
    "tsx",
    "css",
    "scss",
    "sass",
    "less",
    // Systems / other
    "rs",
    "go",
    "swift",
    "sql",
    // Config-ish sources that read as text
    "toml",
    "yaml",
    "yml",
    "json",
    "ini",
    "cfg",
    "properties",
];

/// Source code files. The content is plain text, so extraction is delegated to
/// [`PlainTextParser`]; only the claimed extensions differ.
pub struct CodeParser;

impl DocumentParser for CodeParser {
    fn supports(&self, path: &Path) -> bool {
        match path.extension().and_then(|e| e.to_str()) {
            Some(ext) => CODE_EXTENSIONS.contains(&ext.to_ascii_lowercase().as_str()),
            None => false,
        }
    }

    fn parse(&self, path: &Path) -> Result<String> {
        PlainTextParser.parse(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supports_common_source_extensions() {
        assert!(CodeParser.supports(Path::new("Main.java")));
        assert!(CodeParser.supports(Path::new("script.py")));
        assert!(CodeParser.supports(Path::new("build.gradle")));
        assert!(CodeParser.supports(Path::new("deploy.sh")));
        assert!(CodeParser.supports(Path::new("run.BAT")));
    }

    #[test]
    fn ignores_non_code_files() {
        assert!(!CodeParser.supports(Path::new("report.pdf")));
        assert!(!CodeParser.supports(Path::new("notes.txt")));
        // Extensionless files stay with PlainTextParser.
        assert!(!CodeParser.supports(Path::new("LICENSE")));
    }

    #[test]
    fn reads_file_contents_as_text() {
        let dir = std::env::temp_dir().join("ragcli_code_parser_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("hello.py");
        std::fs::write(&path, "print('hi')\n").unwrap();

        assert_eq!(CodeParser.parse(&path).unwrap(), "print('hi')\n");

        std::fs::remove_file(&path).ok();
    }
}
