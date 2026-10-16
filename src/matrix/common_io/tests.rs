use super::*;
use crate::matrix::parquet::{write_named_table, Column};
use crate::matrix::test_support::{scratch, scratch_gz, write_gz_members};

fn names(v: &[Box<str>]) -> Vec<&str> {
    v.iter().map(std::convert::AsRef::as_ref).collect()
}

#[test]
fn plain_txt_one_name_per_line() {
    let dir = tempfile::tempdir().unwrap();
    let f = scratch(&dir, "genes.txt", "CD8A\nMS4A1\nLYZ\n");
    let got = read_name_list(&f).unwrap();
    assert_eq!(names(&got), ["CD8A", "MS4A1", "LYZ"]);
}

/// The headline case: a curated `gene<TAB>celltype` marker table passed as-is.
/// The celltype column must be ignored, not treated as a second gene.
#[test]
fn tsv_gene_celltype_keeps_only_the_gene_column() {
    let dir = tempfile::tempdir().unwrap();
    let f = scratch(
        &dir,
        "markers.tsv",
        "gene\tcelltype\nCD8A\tT cell\nMS4A1\tB cell\nLYZ\tMonocyte\n",
    );
    let got = read_name_list(&f).unwrap();
    assert_eq!(names(&got), ["CD8A", "MS4A1", "LYZ"]);
}

#[test]
fn csv_without_header_falls_back_to_first_column() {
    let dir = tempfile::tempdir().unwrap();
    let f = scratch(&dir, "markers.csv", "CD8A,T cell\nMS4A1,B cell\n");
    let got = read_name_list(&f).unwrap();
    assert_eq!(names(&got), ["CD8A", "MS4A1"]);
}

/// A gene-like header anywhere picks that column, not column 0.
#[test]
fn header_selects_the_named_column() {
    let dir = tempfile::tempdir().unwrap();
    let f = scratch(
        &dir,
        "markers.tsv",
        "celltype\tsymbol\nT cell\tCD8A\nB cell\tMS4A1\n",
    );
    let got = read_name_list(&f).unwrap();
    assert_eq!(names(&got), ["CD8A", "MS4A1"]);
}

/// Without a recognizable header the first row is data, not a header — losing it
/// would silently drop a gene.
#[test]
fn unrecognized_header_is_treated_as_data() {
    let dir = tempfile::tempdir().unwrap();
    let f = scratch(&dir, "genes.txt", "CD8A\nMS4A1\n");
    assert_eq!(read_name_list(&f).unwrap().len(), 2);
}

#[test]
fn duplicates_and_blank_lines_are_dropped() {
    let dir = tempfile::tempdir().unwrap();
    let f = scratch(&dir, "genes.txt", "CD8A\n\nMS4A1\nCD8A\n   \nLYZ\n");
    let got = read_name_list(&f).unwrap();
    assert_eq!(names(&got), ["CD8A", "MS4A1", "LYZ"]);
}

/// Ragged leading/repeated whitespace must not shift the column.
#[test]
fn whitespace_separated_with_ragged_indent() {
    let dir = tempfile::tempdir().unwrap();
    let f = scratch(&dir, "genes.txt", "  CD8A   T cell\nMS4A1  B cell\n");
    let got = read_name_list(&f).unwrap();
    assert_eq!(names(&got), ["CD8A", "MS4A1"]);
}

#[test]
fn quoted_csv_fields_are_unquoted() {
    let dir = tempfile::tempdir().unwrap();
    let f = scratch(
        &dir,
        "genes.csv",
        "\"gene\",\"celltype\"\n\"CD8A\",\"T cell\"\n",
    );
    let got = read_name_list(&f).unwrap();
    assert_eq!(names(&got), ["CD8A"]);
}

#[test]
fn empty_file_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let f = scratch(&dir, "genes.txt", "\n\n");
    assert!(read_name_list(&f).is_err());
}

/// The parquet path: an all-string table, which the numeric `ParquetReader`
/// cannot read at all.
#[test]
fn parquet_gene_celltype_keeps_only_the_gene_column() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("markers.parquet");
    let path = path.to_string_lossy().into_owned();

    let genes: Vec<Box<str>> = vec!["CD8A".into(), "MS4A1".into(), "LYZ".into()];
    let celltypes: Vec<Box<str>> = vec!["T cell".into(), "B cell".into(), "Monocyte".into()];
    write_named_table(
        &path,
        "gene",
        &genes,
        &[("celltype".into(), Column::Str(&celltypes))],
    )
    .unwrap();

    let got = read_name_list(&path).unwrap();
    assert_eq!(names(&got), ["CD8A", "MS4A1", "LYZ"]);
}

////////////////////////////////////
// First-line header detection     //
////////////////////////////////////

#[test]
fn a_non_numeric_first_line_is_a_header() {
    let dir = tempfile::tempdir().unwrap();
    let f = scratch(&dir, "t.tsv", "gene\ts0\ts1\ng0\t1\t2\n");
    assert_eq!(detect_header_row_numeric(&f, &['\t', ',']), Some(0));
}

#[test]
fn an_all_numeric_first_line_is_data() {
    let dir = tempfile::tempdir().unwrap();
    let f = scratch(&dir, "t.tsv", "g0\t1\t2\ng1\t3\t4\n");
    assert_eq!(detect_header_row_numeric(&f, &['\t', ',']), None);
}

/// A fully quoted numeric field still reads as numeric, or a quoted headerless
/// file loses its first data row to a phantom header.
#[test]
fn quoted_numbers_are_still_numbers() {
    let dir = tempfile::tempdir().unwrap();
    let f = scratch(&dir, "t.csv", "\"g0\",\"1.5\",\"2\"\n");
    assert_eq!(detect_header_row_numeric(&f, &['\t', ',']), None);
}

/// Delimited tables are routinely gzipped; detection has to read through that.
#[test]
fn header_detection_reads_through_gzip() {
    let dir = tempfile::tempdir().unwrap();
    let f = scratch_gz(&dir, "t.tsv.gz", "gene\ts0\ts1\ng0\t1\t2\n");
    assert_eq!(detect_header_row_numeric(&f, &['\t', ',']), Some(0));
}

#[test]
fn first_line_fields_are_unquoted_and_trimmed() {
    let dir = tempfile::tempdir().unwrap();
    let f = scratch(&dir, "t.csv", "\"gene\", s0 ,s1\r\ng0,1,2\n");
    let got = first_line_fields(&f, &['\t', ',']).unwrap();
    assert_eq!(names(&got), ["gene", "s0", "s1"]);
}

/// R writes missing values as `NA`, which is not a number to `f64::from_str`;
/// a headerless file with one in its first row must not lose that row to a
/// phantom header.
#[test]
fn r_missing_values_do_not_make_a_data_row_a_header() {
    let dir = tempfile::tempdir().unwrap();
    let f = scratch(&dir, "t.tsv", "g0\tNA\t2\ng1\t3\tN/A\n");
    assert_eq!(detect_header_row_numeric(&f, &['\t', ',']), None);
}

////////////////////////////////////
// Multi-member gzip (BGZF)        //
////////////////////////////////////

/// `bgzip` output is a concatenation of independent gzip members, each with an
/// `FEXTRA` field. A reader that stops at the end of the first member returns
/// a prefix of the file with no error at all.
#[test]
fn a_multi_member_gzip_is_read_to_the_end() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("names.txt.gz");
    write_gz_members(&path, "CD8A\nMS4A1\nLYZ\nNKG7\nGNLY\n", 3, true);

    let got: Vec<String> = open_buf_reader(path.to_str().unwrap())
        .unwrap()
        .lines()
        .map(|l| l.unwrap())
        .collect();
    assert_eq!(got, ["CD8A", "MS4A1", "LYZ", "NKG7", "GNLY"]);
}

#[test]
fn file_stem_drops_parquet_and_vcf_extensions_too() {
    assert_eq!(
        file_stem("a/Whole_Blood.v10.eQTLs.signif_pairs.parquet"),
        "Whole_Blood.v10.eQTLs.signif_pairs"
    );
    assert_eq!(file_stem("ieu-a-2.vcf.gz"), "ieu-a-2");
    assert_eq!(file_stem("x/goa_human.gaf.gz"), "goa_human");
}

/// A zip in `dir` written by `build`, entry names as given.
fn write_zip(
    dir: &tempfile::TempDir,
    build: impl FnOnce(&mut zip::ZipWriter<File>, zip::write::SimpleFileOptions),
) -> std::path::PathBuf {
    let zip_path = dir.path().join("archive.zip");
    let mut w = zip::ZipWriter::new(File::create(&zip_path).unwrap());
    build(&mut w, zip::write::SimpleFileOptions::default());
    w.finish().unwrap();
    zip_path
}

/// A zip of the given (name, contents) entries.
fn archive_of(dir: &tempfile::TempDir, entries: &[(&str, &[u8])]) -> std::path::PathBuf {
    write_zip(dir, |w, opts| {
        for (name, bytes) in entries {
            w.start_file(*name, opts).unwrap();
            w.write_all(bytes).unwrap();
        }
    })
}

fn unzip_into(zip_path: &Path, target: &Path) -> anyhow::Result<Box<str>> {
    unzip_dir(zip_path.to_str().unwrap(), Some(target.to_str().unwrap()))
}

/// An entry that climbs out of the extraction directory (`../x`, "zip slip")
/// refuses the whole archive, before anything is written.
#[test]
fn unzip_refuses_an_entry_that_leaves_the_extraction_directory() {
    let dir = tempfile::tempdir().unwrap();
    let zip_path = archive_of(
        &dir,
        &[("inside.txt", b"ok"), ("../escape.txt", b"escaped")],
    );
    let target = dir.path().join("out");
    let err = unzip_into(&zip_path, &target).unwrap_err().to_string();
    assert!(err.contains("no safe path"), "{err}");
    assert!(
        !dir.path().join("escape.txt").exists(),
        "written outside the target"
    );
    assert!(
        !target.join("inside.txt").exists(),
        "a refused archive left files behind"
    );
}

/// zip 9.0.3's sanitiser let a verbatim Windows prefix hide `..` from it.
#[test]
fn unzip_refuses_a_verbatim_prefixed_escape() {
    let dir = tempfile::tempdir().unwrap();
    let zip_path = archive_of(
        &dir,
        &[("inside.txt", b"ok"), ("\\\\?\\C:\\../escape.txt", b"x")],
    );
    let target = dir.path().join("out");
    let err = unzip_into(&zip_path, &target).unwrap_err().to_string();
    assert!(err.contains("no safe path"), "{err}");
    assert!(!dir.path().join("escape.txt").exists());
    assert!(
        !target.join("inside.txt").exists(),
        "a refused archive left files behind"
    );
}

/// An absolute name is kept inside the target: its root is dropped.
#[test]
fn unzip_keeps_an_absolute_name_inside_the_target() {
    let dir = tempfile::tempdir().unwrap();
    let zip_path = archive_of(&dir, &[("/abs_probe.txt", b"inside")]);
    let target = dir.path().join("out");
    unzip_into(&zip_path, &target).unwrap();
    assert_eq!(
        std::fs::read(target.join("abs_probe.txt")).unwrap(),
        b"inside"
    );
}

/// A symlink entry is written as a plain file, so an archive can never plant
/// a link for a later entry to pass through.
#[cfg(unix)]
#[test]
fn unzip_writes_a_symlink_entry_as_a_plain_file() {
    let dir = tempfile::tempdir().unwrap();
    let zip_path = write_zip(&dir, |w, opts| {
        w.add_symlink("link", "../../etc", opts).unwrap()
    });
    let target = dir.path().join("out");
    unzip_into(&zip_path, &target).unwrap();
    assert!(!target.join("link").symlink_metadata().unwrap().is_symlink());
}

/// The modes an archive records are not applied, neither to what is
/// extracted nor to the target directory itself.
#[cfg(unix)]
#[test]
fn unzip_ignores_the_archive_modes() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let zip_path = write_zip(&dir, |w, opts| {
        w.add_directory("./", opts.unix_permissions(0o500)).unwrap();
        w.add_directory("d/", opts.unix_permissions(0o555)).unwrap();
        w.start_file("d/f.txt", opts.unix_permissions(0o444))
            .unwrap();
        w.write_all(b"x").unwrap();
    });
    let target = dir.path().join("out");
    std::fs::create_dir_all(&target).unwrap();
    let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
    let before = mode(&target);
    unzip_into(&zip_path, &target).unwrap();
    assert_eq!(mode(&target), before, "the target's mode changed");
    assert_ne!(
        mode(&target.join("d")) & 0o200,
        0,
        "the directory is read-only"
    );
    assert_ne!(
        mode(&target.join("d/f.txt")) & 0o200,
        0,
        "the file is read-only"
    );
}

/// An encrypted entry cannot be read, so the archive is refused up front
/// rather than halfway through.
#[test]
fn unzip_refuses_an_encrypted_entry_before_writing() {
    let dir = tempfile::tempdir().unwrap();
    let zip_path = write_zip(&dir, |w, opts| {
        w.start_file("first.txt", opts).unwrap();
        w.write_all(b"ok").unwrap();
        w.start_file(
            "secret.txt",
            opts.with_aes_encryption(zip::AesMode::Aes256, "pw"),
        )
        .unwrap();
        w.write_all(b"x").unwrap();
    });
    let target = dir.path().join("out");
    let err = unzip_into(&zip_path, &target).unwrap_err().to_string();
    assert!(err.contains("encrypted"), "{err}");
    assert!(
        !target.join("first.txt").exists(),
        "a refused archive left files behind"
    );
}

/// A write that fails names the archive and the path it was writing.
#[test]
fn unzip_errors_name_the_path() {
    let dir = tempfile::tempdir().unwrap();
    let zip_path = archive_of(&dir, &[("d/x.txt", b"x")]);
    let target = dir.path().join("out");
    std::fs::create_dir_all(&target).unwrap();
    std::fs::write(target.join("d"), b"a file where a directory goes").unwrap();
    let err = format!("{:#}", unzip_into(&zip_path, &target).unwrap_err());
    assert!(
        err.contains("archive.zip") && err.contains("d/x.txt"),
        "{err}"
    );
}

/// `unzip_subdir` brings back what `zip_dir_as` zipped, behind a named root
/// or the source's basename, flat; entries outside the root (an OS's
/// metadata, a README) are left out.
#[test]
fn unzip_subdir_unzips_a_zipped_dir_back_flat() {
    for (root, given) in [
        (Some("run"), "run"),
        (Some("run"), "./run"),
        (None, "run.zarr"),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("run.zarr");
        std::fs::create_dir_all(src.join("X/c/0")).unwrap();
        std::fs::write(src.join("zarr.json"), b"{}").unwrap();
        std::fs::write(src.join("X/c/0/1"), b"chunk").unwrap();
        let zip_path = dir.path().join("renamed.zip");
        let zip = zip_path.to_str().unwrap();
        zip_dir_as(src.to_str().unwrap(), zip, root).unwrap();
        let mut w = zip::ZipWriter::new_append(
            std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(zip)
                .unwrap(),
        )
        .unwrap();
        let opts = zip::write::SimpleFileOptions::default();
        w.start_file("README.txt", opts).unwrap();
        w.start_file("__MACOSX/run/._zarr.json", opts).unwrap();
        w.finish().unwrap();
        let out = dir.path().join("back");
        unzip_subdir(zip, given, out.to_str().unwrap()).unwrap();
        assert_eq!(
            std::fs::read(out.join("zarr.json")).unwrap(),
            b"{}",
            "{given}"
        );
        assert_eq!(
            std::fs::read(out.join("X/c/0/1")).unwrap(),
            b"chunk",
            "{given}"
        );
        assert!(!out.join("README.txt").exists() && !out.join("__MACOSX").exists());
    }
}

/// An escape is refused even when it hides behind the root being stripped.
#[test]
fn unzip_subdir_refuses_an_escape_behind_the_root() {
    let dir = tempfile::tempdir().unwrap();
    let zip_path = archive_of(
        &dir,
        &[("run/a.txt", b"ok"), ("run/../../escape.txt", b"x")],
    );
    let target = dir.path().join("out");
    let err = unzip_subdir(zip_path.to_str().unwrap(), "run", target.to_str().unwrap())
        .unwrap_err()
        .to_string();
    assert!(err.contains("no safe path"), "{err}");
    assert!(!dir.path().join("escape.txt").exists());
    assert!(
        !target.join("a.txt").exists(),
        "a refused archive left files behind"
    );
}

/// A root that holds no entry is an error, not an empty success.
#[test]
fn unzip_subdir_refuses_a_root_with_no_entries() {
    let dir = tempfile::tempdir().unwrap();
    let zip_path = archive_of(&dir, &[("run/a.txt", b"ok")]);
    let target = dir.path().join("out");
    let err = unzip_subdir(zip_path.to_str().unwrap(), "rnu", target.to_str().unwrap())
        .unwrap_err()
        .to_string();
    assert!(err.contains("no entries under rnu"), "{err}");
}
