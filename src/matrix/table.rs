//! Row-streaming reader over a named-column table: parquet, or delimited
//! text (`.tsv`, `.csv`, `.txt`, gzip or not), with columns picked by
//! header name.
//!
//! Every cell comes back as a string, whatever the storage type: a parquet
//! `INT64` or `DOUBLE` cell is formatted, a null is `""`. Callers that want a
//! number parse it; callers that want a label get the value rather than a
//! blank. Rows stream, so a multi-gigabyte association table is never held
//! in memory.
//!
//! ```no_run
//! use legume_numeric::matrix::table::TableReader;
//! let t = TableReader::open("pairs.parquet")?;
//! let cols = t.select(&["variant_id", "gene_id", "pval_nominal"])?;
//! for row in t.rows(&cols)? {
//!     let row = row?;
//!     let (variant, gene, p) = (&row[0], &row[1], &row[2]);
//! #   let _ = (variant, gene, p);
//! }
//! # Ok::<(), anyhow::Error>(())
//! ```

use crate::matrix::common_io::{open_buf_reader, unquote_field};
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::reader::RowIter;
use parquet::record::Field;
use parquet::schema::types::Type as SchemaType;
use std::fs::File;
use std::io::BufRead;
use std::path::Path;
use std::sync::Arc;

/// How the cells of a text line are separated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Delimiter {
    /// One tab per boundary; empty cells are kept.
    Tab,
    /// One comma per boundary; empty cells are kept, quotes stripped.
    Comma,
    /// Runs of spaces or tabs; empty cells cannot occur.
    Whitespace,
}

impl Delimiter {
    /// The delimiter a header line uses: a tab if it has one, else a comma,
    /// else whitespace.
    pub fn detect(header: &str) -> Self {
        if header.contains('\t') {
            Delimiter::Tab
        } else if header.contains(',') {
            Delimiter::Comma
        } else {
            Delimiter::Whitespace
        }
    }

    /// Split one line into cells, trimmed and unquoted.
    pub fn split<'a>(&self, line: &'a str) -> Vec<&'a str> {
        match self {
            Delimiter::Tab => line.split('\t').map(|s| unquote_field(s.trim())).collect(),
            Delimiter::Comma => line.split(',').map(|s| unquote_field(s.trim())).collect(),
            Delimiter::Whitespace => line.split_whitespace().map(unquote_field).collect(),
        }
    }
}

enum Source {
    Parquet,
    Text { delim: Delimiter },
}

/// A table opened for streaming: its header is read up front, its rows on
/// demand through [`TableReader::rows`].
pub struct TableReader {
    path: Box<str>,
    header: Vec<Box<str>>,
    source: Source,
}

/// `true` when the path names a parquet file (`.parquet` / `.pq`).
pub fn is_parquet_path(path: &str) -> bool {
    matches!(
        Path::new(path)
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("parquet") | Some("pq")
    )
}

impl TableReader {
    /// Open `path` and read its header. Parquet is recognised by extension;
    /// anything else is delimited text, gzip-decoded when it ends in `.gz`.
    /// A text header is the first line that does not start with `##` (VCF-
    /// style meta lines are skipped), with one leading `#` dropped
    /// (`#chr start end` is a BED-style header).
    pub fn open(path: &str) -> anyhow::Result<Self> {
        if is_parquet_path(path) {
            let file = File::open(path).map_err(|e| anyhow::anyhow!("opening {path}: {e}"))?;
            let reader = SerializedFileReader::new(file)?;
            let header = reader
                .metadata()
                .file_metadata()
                .schema()
                .get_fields()
                .iter()
                .map(|f| Box::from(f.name()))
                .collect();
            return Ok(Self {
                path: path.into(),
                header,
                source: Source::Parquet,
            });
        }
        let mut reader =
            open_buf_reader(path).map_err(|e| anyhow::anyhow!("opening {path}: {e}"))?;
        let mut line = String::new();
        anyhow::ensure!(
            read_header_line(&mut reader, &mut line)?,
            "{path}: no header line"
        );
        let l = line.trim_end_matches(['\n', '\r']);
        let l = l.strip_prefix('#').unwrap_or(l);
        let delim = Delimiter::detect(l);
        let header = delim.split(l).into_iter().map(Box::from).collect();
        Ok(Self {
            path: path.into(),
            header,
            source: Source::Text { delim },
        })
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    /// Column names, in file order.
    pub fn header(&self) -> &[Box<str>] {
        &self.header
    }

    /// Index of a column by exact name.
    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.header.iter().position(|h| h.as_ref() == name)
    }

    /// `true` when every name is a column.
    pub fn has_columns(&self, names: &[&str]) -> bool {
        names.iter().all(|n| self.column_index(n).is_some())
    }

    /// The first of `candidates` that is a column: for a field that
    /// different releases spell differently.
    pub fn find_column(&self, candidates: &[&str]) -> Option<usize> {
        candidates.iter().find_map(|c| self.column_index(c))
    }

    /// Indices of `names`, in request order; a missing name is an error
    /// that lists what the file does have.
    pub fn select(&self, names: &[&str]) -> anyhow::Result<Vec<usize>> {
        names
            .iter()
            .map(|n| {
                self.column_index(n).ok_or_else(|| {
                    anyhow::anyhow!(
                        "{}: no column `{n}`; the header has {}",
                        self.path,
                        self.header.join(", ")
                    )
                })
            })
            .collect()
    }

    /// Stream the rows, each cut down to the columns at `cols` (in that
    /// order; repeats allowed). A text row too short for a requested column
    /// yields `""` there; blank lines and `#` lines are skipped.
    pub fn rows(&self, cols: &[usize]) -> anyhow::Result<TableRows> {
        if let Some(&j) = cols.iter().find(|&&j| j >= self.header.len()) {
            anyhow::bail!(
                "{}: column index {j} out of range ({} columns)",
                self.path,
                self.header.len()
            );
        }
        match &self.source {
            Source::Parquet => self.parquet_rows(cols),
            Source::Text { delim } => {
                let mut reader = open_buf_reader(&self.path)?;
                let mut line = String::new();
                read_header_line(&mut reader, &mut line)?;
                Ok(TableRows::Text {
                    reader,
                    delim: *delim,
                    cols: cols.to_vec(),
                    line: String::new(),
                })
            }
        }
    }

    fn parquet_rows(&self, cols: &[usize]) -> anyhow::Result<TableRows> {
        let file = File::open(&*self.path)?;
        let reader = SerializedFileReader::new(file)?;
        let root = reader.metadata().file_metadata().schema();
        let root_name = root.name().to_string();
        let fields = root.get_fields().to_vec();
        // Project onto the distinct requested columns in file order, then map
        // every request onto its position in the projection.
        let mut distinct: Vec<usize> = cols.to_vec();
        distinct.sort_unstable();
        distinct.dedup();
        let proj_fields: Vec<Arc<SchemaType>> =
            distinct.iter().map(|&j| fields[j].clone()).collect();
        let proj = SchemaType::group_type_builder(&root_name)
            .with_fields(proj_fields)
            .build()?;
        let pos: Vec<usize> = cols
            .iter()
            .map(|j| distinct.binary_search(j).expect("in the projection"))
            .collect();
        let iter = RowIter::from_file_into(Box::new(reader)).project(Some(proj))?;
        Ok(TableRows::Parquet {
            iter: Box::new(iter),
            pos,
        })
    }
}

/// Read up to and including the header line: the first non-blank line not
/// starting with `##`, left in `line`. `false` at end of file.
fn read_header_line(reader: &mut dyn BufRead, line: &mut String) -> anyhow::Result<bool> {
    loop {
        line.clear();
        if reader.read_line(line)? == 0 {
            return Ok(false);
        }
        let l = line.trim_end_matches(['\n', '\r']);
        if !l.starts_with("##") && !l.trim().is_empty() {
            return Ok(true);
        }
    }
}

/// The row stream of a [`TableReader`].
pub enum TableRows {
    Parquet {
        iter: Box<RowIter<'static>>,
        pos: Vec<usize>,
    },
    Text {
        reader: Box<dyn BufRead>,
        delim: Delimiter,
        cols: Vec<usize>,
        line: String,
    },
}

impl Iterator for TableRows {
    type Item = anyhow::Result<Vec<Box<str>>>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            TableRows::Parquet { iter, pos } => {
                let row = match iter.next()? {
                    Ok(r) => r,
                    Err(e) => return Some(Err(e.into())),
                };
                let cells: Vec<&Field> = row.get_column_iter().map(|(_, f)| f).collect();
                Some(Ok(pos.iter().map(|&p| field_to_string(cells[p])).collect()))
            }
            TableRows::Text {
                reader,
                delim,
                cols,
                line,
            } => loop {
                line.clear();
                match reader.read_line(line) {
                    Ok(0) => return None,
                    Ok(_) => {}
                    Err(e) => return Some(Err(e.into())),
                }
                let l = line.trim_end_matches(['\n', '\r']);
                if l.trim().is_empty() || l.starts_with('#') {
                    continue;
                }
                let cells = delim.split(l);
                return Some(Ok(cols
                    .iter()
                    .map(|&j| Box::from(cells.get(j).copied().unwrap_or("")))
                    .collect()));
            },
        }
    }
}

/// A parquet cell as text: strings verbatim, numbers formatted, null `""`.
pub fn field_to_string(f: &Field) -> Box<str> {
    match f {
        Field::Null => "".into(),
        Field::Str(s) => s.as_str().into(),
        Field::Bool(b) => b.to_string().into(),
        Field::Byte(x) => x.to_string().into(),
        Field::Short(x) => x.to_string().into(),
        Field::Int(x) => x.to_string().into(),
        Field::Long(x) => x.to_string().into(),
        Field::UByte(x) => x.to_string().into(),
        Field::UShort(x) => x.to_string().into(),
        Field::UInt(x) => x.to_string().into(),
        Field::ULong(x) => x.to_string().into(),
        Field::Float(x) => x.to_string().into(),
        Field::Double(x) => x.to_string().into(),
        Field::Float16(x) => x.to_string().into(),
        Field::Bytes(b) => String::from_utf8_lossy(b.data()).into_owned().into(),
        other => other.to_string().into(),
    }
}

#[cfg(test)]
#[path = "table_tests.rs"]
mod tests;
