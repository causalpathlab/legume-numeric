//! One table, three encodings: tsv, tsv.gz and parquet read back the same
//! cells, numeric parquet cells included.

use super::*;
use crate::matrix::parquet::{write_table, Column};
use std::io::Write;

fn boxed(v: &[&str]) -> Vec<Box<str>> {
    v.iter().map(|s| Box::from(*s)).collect()
}

const TSV: &str = "variant_id\tgene_id\tpval\tn\n\
                   chr1_100_A_G_b38\tENSG01.3\t0.001\t7\n\
                   chr2_200_C_T_b38\t\t0.5\t9\n";

fn collect(path: &str, names: &[&str]) -> Vec<Vec<Box<str>>> {
    let t = TableReader::open(path).unwrap();
    let cols = t.select(names).unwrap();
    t.rows(&cols).unwrap().map(Result::unwrap).collect()
}

#[test]
fn tsv_gz_and_parquet_stream_the_same_selected_cells() {
    let dir = tempfile::tempdir().unwrap();
    let tsv = dir.path().join("t.tsv");
    std::fs::write(&tsv, TSV).unwrap();
    let gz = dir.path().join("t.tsv.gz");
    let mut enc = flate2::write::GzEncoder::new(
        std::fs::File::create(&gz).unwrap(),
        flate2::Compression::default(),
    );
    enc.write_all(TSV.as_bytes()).unwrap();
    enc.finish().unwrap();
    let pq = dir.path().join("t.parquet");
    write_table(
        pq.to_str().unwrap(),
        &[
            (
                "variant_id".into(),
                Column::Str(&boxed(&["chr1_100_A_G_b38", "chr2_200_C_T_b38"])),
            ),
            ("gene_id".into(), Column::Str(&boxed(&["ENSG01.3", ""]))),
            ("pval".into(), Column::F32(&[0.001, 0.5])),
            ("n".into(), Column::I64(&[7, 9])),
        ],
    )
    .unwrap();

    // Out of file order, with a repeat, to exercise the projection mapping.
    let want = ["n", "variant_id", "gene_id", "pval", "n"];
    let a = collect(tsv.to_str().unwrap(), &want);
    let b = collect(gz.to_str().unwrap(), &want);
    let c = collect(pq.to_str().unwrap(), &want);
    assert_eq!(a, b);
    assert_eq!(a, c, "parquet numbers are stringified, not blanked");
    assert_eq!(
        a[0],
        boxed(&["7", "chr1_100_A_G_b38", "ENSG01.3", "0.001", "7"])
    );
    assert_eq!(a[1][2].as_ref(), "", "an empty text cell is kept as empty");
}

#[test]
fn headers_skip_meta_lines_and_a_leading_hash_and_missing_columns_are_named() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("b.bed");
    std::fs::write(
        &p,
        "##meta\n#chr\tstart\tend\n1\t10\t20\n# comment\n2\t30\t40\n",
    )
    .unwrap();
    let t = TableReader::open(p.to_str().unwrap()).unwrap();
    assert_eq!(t.header(), &boxed(&["chr", "start", "end"])[..]);
    assert_eq!(t.find_column(&["begin", "start"]), Some(1));
    let rows: Vec<_> = t.rows(&[0, 2]).unwrap().map(Result::unwrap).collect();
    assert_eq!(rows, vec![boxed(&["1", "20"]), boxed(&["2", "40"])]);
    let err = t.select(&["pip"]).unwrap_err().to_string();
    assert!(
        err.contains("pip") && err.contains("chr, start, end"),
        "{err}"
    );

    let csv = dir.path().join("c.csv");
    std::fs::write(&csv, "a,b\n\"x\",1\n").unwrap();
    assert_eq!(
        collect(csv.to_str().unwrap(), &["b", "a"]),
        vec![boxed(&["1", "x"])]
    );
}
