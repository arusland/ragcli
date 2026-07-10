/// Splits text into chunks of at most `chunk_size` characters with `overlap`
/// characters carried over between consecutive chunks. Prefers to break at a
/// paragraph boundary, then at a line break, then at whitespace.
pub fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    assert!(chunk_size > 0, "chunk_size must be positive");
    assert!(
        overlap < chunk_size,
        "overlap must be smaller than chunk_size"
    );

    let chars: Vec<char> = text.chars().collect();
    let mut chunks = Vec::new();
    let mut start = 0;

    while start < chars.len() {
        let hard_end = (start + chunk_size).min(chars.len());
        let end = if hard_end == chars.len() {
            hard_end
        } else {
            find_break(&chars, start, hard_end)
        };

        let chunk: String = chars[start..end].iter().collect();
        if !chunk.trim().is_empty() {
            chunks.push(chunk.trim().to_string());
        }

        if end == chars.len() {
            break;
        }
        start = end.saturating_sub(overlap).max(start + 1);
    }

    chunks
}

/// Picks a break position in `(start, hard_end]`, preferring paragraph breaks,
/// then newlines, then whitespace within the last portion of the window.
fn find_break(chars: &[char], start: usize, hard_end: usize) -> usize {
    // Only look back over the tail of the window so chunks stay near chunk_size.
    let search_from = start + (hard_end - start) * 2 / 3;

    let mut newline_at = None;
    let mut space_at = None;
    let mut i = hard_end;
    while i > search_from {
        i -= 1;
        match chars[i] {
            '\n' => {
                if i > 0 && chars[i - 1] == '\n' {
                    return i + 1; // paragraph boundary
                }
                newline_at.get_or_insert(i + 1);
            }
            c if c.is_whitespace() => {
                space_at.get_or_insert(i + 1);
            }
            _ => {}
        }
    }
    newline_at.or(space_at).unwrap_or(hard_end)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_yields_no_chunks() {
        assert!(chunk_text("", 100, 10).is_empty());
        assert!(chunk_text("   \n\n  ", 100, 10).is_empty());
    }

    #[test]
    fn short_input_yields_single_chunk() {
        let chunks = chunk_text("hello world", 100, 10);
        assert_eq!(chunks, vec!["hello world"]);
    }

    #[test]
    fn long_input_is_split_with_overlap() {
        let text = "word ".repeat(200); // 1000 chars
        let chunks = chunk_text(&text, 100, 20);
        assert!(chunks.len() > 1);
        for chunk in &chunks {
            assert!(chunk.chars().count() <= 100);
        }
        // Consecutive chunks share overlapping content.
        let full: String = chunks.join("");
        assert!(full.len() >= text.trim().len() - chunks.len());
    }

    #[test]
    fn prefers_paragraph_boundary() {
        let text = format!("{}\n\n{}", "a".repeat(80), "b".repeat(80));
        let chunks = chunk_text(&text, 100, 0);
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].chars().all(|c| c == 'a'));
        assert!(chunks[1].chars().all(|c| c == 'b'));
    }

    #[test]
    fn always_makes_progress() {
        // overlap close to chunk_size must not loop forever
        let text = "x".repeat(50);
        let chunks = chunk_text(&text, 10, 9);
        assert!(!chunks.is_empty());
    }
}
