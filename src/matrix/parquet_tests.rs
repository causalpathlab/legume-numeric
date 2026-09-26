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
