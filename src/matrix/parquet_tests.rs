//! The name column of a parquet table is found by TYPE, not by position: the
//! first string column. A table written without its index has none, and
//! saying so beats stringifying the first sample's counts into "gene names".

use super::*;
use crate::matrix::traits::IoOps;

fn labels(v: &[&str]) -> Vec<Box<str>> {
    v.iter().map(|s| Box::from(*s)).collect()
}

#[test]
fn the_name_column_is_the_first_string_column_wherever_it_sits() {
    let dir = tempfile::tempdir().unwrap();
    let f = dir.path().join("t.parquet");
    let f = f.to_str().unwrap();
    let genes = labels(&["g0", "g1"]);
    write_table(
        f,
        &[
            ("s0".into(), Column::F32(&[1.0, 2.0])),
            ("gene".into(), Column::Str(&genes)),
            ("s1".into(), Column::F32(&[3.0, 4.0])),
        ],
    )
    .unwrap();
    assert_eq!(first_string_column(f).unwrap(), Some(1));
}

#[test]
fn a_table_without_strings_has_no_name_column() {
    let dir = tempfile::tempdir().unwrap();
    let f = dir.path().join("t.parquet");
    let f = f.to_str().unwrap();
    write_table(
        f,
        &[
            ("s0".into(), Column::F32(&[1.0, 2.0])),
            ("s1".into(), Column::F32(&[3.0, 4.0])),
        ],
    )
    .unwrap();
    assert_eq!(first_string_column(f).unwrap(), None);
}

/// `write_table` is the keyless sibling of `write_named_table`; the matrix
/// reader must see the same thing through both.
#[test]
fn write_table_round_trips_through_the_matrix_reader() {
    let dir = tempfile::tempdir().unwrap();
    let f = dir.path().join("t.parquet");
    let f = f.to_str().unwrap();
    let genes = labels(&["g0", "g1"]);
    write_table(
        f,
        &[
            ("gene".into(), Column::Str(&genes)),
            ("s0".into(), Column::F32(&[1.0, 2.0])),
            ("s1".into(), Column::F32(&[3.0, 4.0])),
        ],
    )
    .unwrap();
    let m = nalgebra::DMatrix::<f32>::from_parquet(f).unwrap();
    assert_eq!(m.rows, genes);
    assert_eq!(m.cols, labels(&["s0", "s1"]));
    assert_eq!(
        m.mat,
        nalgebra::DMatrix::<f32>::from_row_slice(2, 2, &[1.0, 3.0, 2.0, 4.0])
    );
}

#[test]
fn string_columns_by_name_format_numeric_cells_instead_of_blanking_them() {
    let dir = tempfile::tempdir().unwrap();
    let f = dir.path().join("t.parquet");
    let f = f.to_str().unwrap();
    let genes = labels(&["g0", "g1"]);
    write_table(
        f,
        &[
            ("gene".into(), Column::Str(&genes)),
            ("cluster".into(), Column::I32(&[3, 11])),
            ("score".into(), Column::F32(&[0.5, 2.0])),
        ],
    )
    .unwrap();
    let cols = read_parquet_string_columns_by_name(f, &["cluster", "gene", "score"]).unwrap();
    assert_eq!(cols[0], labels(&["3", "11"]));
    assert_eq!(cols[1], genes);
    assert_eq!(cols[2], labels(&["0.5", "2"]));
}

/// A small `row, a, b, c` table: one string name column, three numeric ones.
fn abc_table(dir: &tempfile::TempDir) -> String {
    let f = dir.path().join("t.parquet");
    let f = f.to_str().unwrap().to_string();
    let rows = labels(&["r0", "r1"]);
    write_table(
        &f,
        &[
            ("row".into(), Column::Str(&rows)),
            ("a".into(), Column::F32(&[1.0, 2.0])),
            ("b".into(), Column::F32(&[3.0, 4.0])),
            ("c".into(), Column::F32(&[5.0, 6.0])),
        ],
    )
    .unwrap();
    f
}

/// Asking for columns by name that the file does not have is an error, not a
/// silent read of every other column.
#[test]
fn names_that_match_nothing_are_an_error_naming_what_is_there() {
    let dir = tempfile::tempdir().unwrap();
    let f = abc_table(&dir);
    let err = ParquetReader::new(&f, Some(0), None, Some(&labels(&["x", "y"])))
        .err()
        .expect("unmatched names must not fall back to all columns")
        .to_string();
    for want in ["x", "y", "row", "a", "b", "c"] {
        assert!(err.contains(want), "error {err:?} should mention {want:?}");
    }
}

/// Names are candidates: callers pass a list covering several layouts at
/// once, so an absent name is skipped as long as one is present.
#[test]
fn absent_names_are_skipped_when_another_matches() {
    let dir = tempfile::tempdir().unwrap();
    let f = abc_table(&dir);
    let r = ParquetReader::new(&f, Some(0), None, Some(&labels(&["zz", "b"]))).unwrap();
    assert_eq!(r.column_names, labels(&["b"]));
    assert_eq!(r.row_major_data, vec![3.0, 4.0]);
}

#[test]
fn a_row_name_index_past_the_last_column_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let f = abc_table(&dir);
    let err = ParquetReader::new(&f, Some(7), None, None)
        .err()
        .expect("an out-of-range row-name index must not panic")
        .to_string();
    assert!(err.contains("row-name") && err.contains('7'), "{err}");
}

#[test]
fn matched_names_select_only_those_columns() {
    let dir = tempfile::tempdir().unwrap();
    let f = abc_table(&dir);
    let r = ParquetReader::new(&f, Some(0), None, Some(&labels(&["c", "a"]))).unwrap();
    assert_eq!(r.column_names, labels(&["a", "c"]));
    assert_eq!(r.row_major_data, vec![1.0, 5.0, 2.0, 6.0]);
}

/// No selector, or only empty ones, still reads every numeric column.
#[test]
fn no_selector_reads_every_numeric_column() {
    let dir = tempfile::tempdir().unwrap();
    let f = abc_table(&dir);
    let all = labels(&["a", "b", "c"]);
    let r = ParquetReader::new(&f, Some(0), None, None).unwrap();
    assert_eq!(r.column_names, all);
    let r = ParquetReader::new(&f, Some(0), Some(&[]), Some(&[])).unwrap();
    assert_eq!(r.column_names, all);
}

/// Explicit indices override names, as in the delimited-text reader: callers
/// pass their default names alongside user indices, and those defaults need
/// not exist in the file nor widen the selection when they do.
#[test]
fn indices_override_names() {
    let dir = tempfile::tempdir().unwrap();
    let f = abc_table(&dir);
    let r = ParquetReader::new(&f, Some(0), Some(&[2]), Some(&labels(&["x"]))).unwrap();
    assert_eq!(r.column_names, labels(&["b"]));
    let r = ParquetReader::new(&f, Some(0), Some(&[2]), Some(&labels(&["a"]))).unwrap();
    assert_eq!(r.column_names, labels(&["b"]));
}

#[test]
fn an_index_past_the_last_column_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let f = abc_table(&dir);
    let err = ParquetReader::new(&f, Some(0), Some(&[1, 9]), None)
        .err()
        .expect("an out-of-range index must not be dropped")
        .to_string();
    assert!(err.contains('9'), "{err}");
}
