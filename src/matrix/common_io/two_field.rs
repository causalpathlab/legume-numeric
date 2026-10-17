//! Two-column tables of names (`gene<TAB>label`, `label<TAB>id`): one line
//! rule for every reader of marker panels, membership lists and label maps.

use super::{open_buf_reader, unquote_field};
use anyhow::Context;
use std::borrow::Cow;
use std::io::BufRead;

/// The rows of a two-column table. Each line splits by one rule: a line with
/// a tab outside double quotes (trailing whitespace aside) splits on its tabs
/// only, so a cell keeps its commas (`CD4-positive, alpha-beta T cell`); a
/// line without one is CSV, split on commas except inside a double-quoted
/// cell. In a double-quoted cell, on either kind of line, `""` stands for a
/// quote, and text after the closing quote is kept, as CSV readers keep it.
/// Cells keep their columns (an empty one stays empty) and are unquoted and
/// trimmed, inside the quotes too; single quotes unquote a cell
/// ([`unquote_field`]) but never group commas, so an apostrophe stays text. A
/// leading byte order mark is dropped, and an unclosed quote runs to the end
/// of the line, as written.
///
/// Blank lines and comments (a first cell starting with `#` or `%`, indented
/// or quoted) are skipped; so are rows with an empty or missing first or
/// second cell, counted in `gapped`. Cells past the second are not read.
/// Nothing is logged: the caller words its own warnings and drops its own
/// header.
#[derive(Debug, Default)]
#[non_exhaustive]
pub struct TwoFieldRows {
    /// `(first, second)` cells of every row that has both.
    pub rows: Vec<(Box<str>, Box<str>)>,
    /// Rows skipped for an empty or missing first or second cell.
    pub gapped: usize,
    /// CSV rows (no tab) with text in a cell past the second, read for their
    /// first two: often a label holding an unquoted comma.
    pub wide_comma: usize,
}

impl TwoFieldRows {
    /// The rows of a table already in memory.
    pub fn from_text(text: &str) -> Self {
        let mut out = Self::default();
        text.lines().for_each(|line| out.push_line(line));
        out
    }

    fn push_line(&mut self, line: &str) {
        let line = split_line(line);
        match (line.first, line.second) {
            (first, None) if first.is_empty() => {}
            (first, _) if first.starts_with(['#', '%']) => {}
            (first, Some(second)) if !first.is_empty() && !second.is_empty() => {
                if line.wide {
                    self.wide_comma += 1;
                }
                self.rows.push((Box::from(&*first), Box::from(&*second)));
            }
            _ => self.gapped += 1,
        }
    }
}

/// [`TwoFieldRows`] of a file, plain or gzipped. A line that cannot be read
/// (not UTF-8, a corrupt gz) is an error naming the file and line.
pub fn read_two_field_rows(path: &str) -> anyhow::Result<TwoFieldRows> {
    let reader = open_buf_reader(path).with_context(|| format!("opening {path}"))?;
    let mut out = TwoFieldRows::default();
    for (i, line) in reader.lines().enumerate() {
        out.push_line(&line.with_context(|| format!("{path}: line {}", i + 1))?);
    }
    Ok(out)
}

/// A line's first two cells by the [`TwoFieldRows`] rule.
struct Line<'a> {
    first: Cow<'a, str>,
    /// `None` when the line has a single cell.
    second: Option<Cow<'a, str>>,
    /// A CSV line with text in a cell past the second.
    wide: bool,
}

fn split_line(line: &str) -> Line<'_> {
    let line = line.strip_prefix('\u{feff}').unwrap_or(line).trim_end();
    if has_bare_tab(line) {
        let mut cells = line.splitn(3, '\t');
        return Line {
            first: unquote_cell(cells.next().unwrap_or("")),
            second: cells.next().map(unquote_cell),
            wide: false,
        };
    }
    let (first, rest) = csv_cell(line);
    let Some(rest) = rest else {
        return Line {
            first,
            second: None,
            wide: false,
        };
    };
    let (second, rest) = csv_cell(rest);
    // Past the second cell, only text counts: `a,b,,` has none.
    let wide =
        rest.is_some_and(|r| r.contains(|c: char| c != ',' && c != '"' && !c.is_whitespace()));
    Line {
        first,
        second: Some(second),
        wide,
    }
}

/// Whether `line` has a tab outside double quotes.
fn has_bare_tab(line: &str) -> bool {
    let mut quoted = false;
    for c in line.chars() {
        match c {
            '"' => quoted = !quoted,
            '\t' if !quoted => return true,
            _ => {}
        }
    }
    false
}

/// The CSV cell at the start of `line`, and the rest of the line after the
/// comma that ends it (`None` at the end of the line).
fn csv_cell(line: &str) -> (Cow<'_, str>, Option<&str>) {
    let cell = line.trim_start();
    if let Some(body) = cell.strip_prefix('"') {
        if let Some(close) = closing_quote(body) {
            let (tail, next) = match body[close + 1..].split_once(',') {
                Some((tail, next)) => (tail, Some(next)),
                None => (&body[close + 1..], None),
            };
            let text = unescape(&body[..close]);
            let tail = tail.trim_end();
            let text = if tail.trim_start().is_empty() {
                text
            } else {
                Cow::Owned(format!("{text}{tail}").trim().to_owned())
            };
            return (text, next);
        }
        return (Cow::Borrowed(cell), None); // unclosed: as written
    }
    match cell.split_once(',') {
        Some((c, next)) => (unquote_cell(c), Some(next)),
        None => (unquote_cell(cell), None),
    }
}

/// The byte offset in `body` (after an opening quote) of the quote that
/// closes it: the first `"` not doubled.
fn closing_quote(body: &str) -> Option<usize> {
    let mut from = 0;
    loop {
        let at = from + body[from..].find('"')?;
        if body[at + 1..].starts_with('"') {
            from = at + 2;
        } else {
            return Some(at);
        }
    }
}

/// A cell with no grouping quotes: trimmed and unquoted, a double-quoted one
/// with its `""` read as `"`.
fn unquote_cell(cell: &str) -> Cow<'_, str> {
    let t = cell.trim();
    match t.strip_prefix('"').and_then(|t| t.strip_suffix('"')) {
        Some(inner) => unescape(inner),
        None => Cow::Borrowed(unquote_field(t).trim()),
    }
}

/// Quoted text with `""` read as `"`, trimmed.
fn unescape(inner: &str) -> Cow<'_, str> {
    let inner = inner.trim();
    if inner.contains("\"\"") {
        Cow::Owned(inner.replace("\"\"", "\""))
    } else {
        Cow::Borrowed(inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::test_support::{scratch, scratch_gz};

    /// A line's first two cells, and whether it is a wide CSV row.
    fn cells(line: &str) -> (Vec<String>, bool) {
        let l = split_line(line);
        let cells = std::iter::once(l.first).chain(l.second);
        (cells.map(Cow::into_owned).collect(), l.wide)
    }

    fn two(line: &str) -> Vec<String> {
        cells(line).0
    }

    #[test]
    fn a_tab_line_splits_on_its_tabs_only_and_keeps_empty_cells() {
        assert_eq!(
            two("CD4\tCD4-positive, alpha-beta T cell\t\tnote \r"),
            ["CD4", "CD4-positive, alpha-beta T cell"]
        );
        assert_eq!(two("\tT cell"), ["", "T cell"]);
        assert_eq!(two("MS4A1,B cell\t"), ["MS4A1", "B cell"], "trailing tab");
        assert_eq!(two("CD4\t\"a \"\"b\"\" c\""), ["CD4", r#"a "b" c"#]);
    }

    #[test]
    fn a_comma_line_keeps_a_quoted_comma_in_its_cell() {
        assert_eq!(
            cells(r#"CD4,"CD4-positive, alpha-beta T cell",x"#),
            (
                vec!["CD4".into(), "CD4-positive, alpha-beta T cell".into()],
                true
            )
        );
        assert_eq!(two(r#" "a ""b"" c" , d"#), [r#"a "b" c"#, "d"]);
        assert_eq!(two(r#"CD4,"T cell" extra,x"#), ["CD4", "T cell extra"]);
        assert_eq!(two("CD4,\"naive\tT cell\""), ["CD4", "naive\tT cell"]);
        assert_eq!(
            cells("GENE,CT3, sub b"),
            (vec!["GENE".into(), "CT3".into()], true)
        );
    }

    #[test]
    fn quotes_unquote_and_trim_inside_but_single_ones_never_group() {
        assert_eq!(two("'MS4A1',' B cell'"), ["MS4A1", "B cell"]);
        assert_eq!(two("\"T cell \"\tCL:0000084"), ["T cell", "CL:0000084"]);
        assert_eq!(two("RLBP1,Muller's glia"), ["RLBP1", "Muller's glia"]);
    }

    #[test]
    fn a_byte_order_mark_and_an_unclosed_quote_are_read_leniently() {
        assert_eq!(two("\u{feff}gene,celltype"), ["gene", "celltype"]);
        assert_eq!(two(r#"CD4,"T cell, open"#), ["CD4", r#""T cell, open"#]);
    }

    #[test]
    fn rows_skip_blanks_and_comments_and_count_gaps_and_wide_comma_rows() {
        let got = TwoFieldRows::from_text(
            "# note\n  % note\n\n \n\"#x\",y\nLONELY\n\tT cell\nCD3E,\nCD14,Monocyte,,\n\
             CD8A,T cell, CD8\nCD4\tT\tx\nCD2,T,x\t\n",
        );
        let rows: Vec<(&str, &str)> = got.rows.iter().map(|(a, b)| (&**a, &**b)).collect();
        assert_eq!(
            rows,
            [
                ("CD14", "Monocyte"),
                ("CD8A", "T cell"),
                ("CD4", "T"),
                ("CD2", "T")
            ],
            "a first cell starting with # or %, quoted or indented, is a comment"
        );
        assert_eq!(got.gapped, 3, "LONELY, the empty gene and the empty label");
        assert_eq!(
            got.wide_comma, 2,
            "a comma row's extra text is counted (a trailing tab aside), not empty \
             trailing cells nor a tab row's extra cells"
        );
    }

    #[test]
    fn rows_read_through_gzip_and_name_an_unreadable_line() {
        let dir = tempfile::tempdir().unwrap();
        let gz = scratch_gz(&dir, "p.tsv.gz", "CD3E\tT cell\n");
        let rows = read_two_field_rows(&gz).unwrap().rows;
        assert_eq!(rows, [(Box::from("CD3E"), Box::from("T cell"))]);
        let plain = scratch(&dir, "p.tsv", "CD3E\tT cell\n");
        assert_eq!(read_two_field_rows(&plain).unwrap().rows, rows);

        let bad = dir.path().join("bad.tsv");
        std::fs::write(&bad, b"CD3E\tT cell\nCD14\tMono\xffcyte\n").unwrap();
        let err = format!(
            "{:#}",
            read_two_field_rows(bad.to_str().unwrap()).unwrap_err()
        );
        assert!(err.contains("bad.tsv: line 2"), "{err}");
    }
}
