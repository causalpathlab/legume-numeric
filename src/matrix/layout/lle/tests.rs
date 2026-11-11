use super::*;
use crate::matrix::landmark_init::tests::{arranged, clusters, swiss_roll, unrolled};
use crate::matrix::test_support::{fuzzy_edges, pca_start};

/// The `n × 2` layout as `[x0, y0, x1, y1, …]`, as the fixtures read it.
fn flat(y: &Mat) -> Vec<f32> {
    y.transpose().as_slice().to_vec()
}

/// Each coordinate of `sheet` centred and scaled to unit variance.
fn standardized(sheet: &[[f32; 2]]) -> Vec<[f32; 2]> {
    let n = sheet.len() as f32;
    let (mut mean, mut var) = ([0.0_f32; 2], [0.0_f32; 2]);
    for p in sheet {
        (0..2).for_each(|c| mean[c] += p[c] / n);
    }
    for p in sheet {
        (0..2).for_each(|c| var[c] += (p[c] - mean[c]).powi(2) / n);
    }
    sheet
        .iter()
        .map(|p| [0, 1].map(|c| (p[c] - mean[c]) / var[c].sqrt()))
        .collect()
}

#[test]
fn a_swiss_roll_is_unrolled_from_its_graph() {
    // Dense enough that no kNN edge jumps a turn: what LLE makes of a
    // clean graph, not of a shortcut. LLE's axes come out of unit variance,
    // so the sheet is compared with each axis standardized.
    for seed in 1..=2 {
        let (data, sheet) = swiss_roll(1500, seed);
        let y = lle_layout_2d_on_graph(&data, &fuzzy_edges(&data, 10));
        let sheet = standardized(&sheet);
        let lle = unrolled(&sheet, &flat(&y));
        let pca = unrolled(&sheet, &pca_start(&data));
        assert!(
            lle > 0.9 && lle > pca + 0.3,
            "seed {seed}: LLE {lle}, PCA {pca}"
        );
    }
}

#[test]
fn separate_groups_are_arranged_apart_and_none_collapses() {
    // No kNN edge joins two clusters: each is a piece of its own.
    let (data, labels) = clusters(900, 21);
    let y = lle_layout_2d_on_graph(&data, &fuzzy_edges(&data, 15));
    let pca = arranged(&data, &labels, &pca_start(&data));
    let lle = arranged(&data, &labels, &flat(&y));
    assert!(lle > pca - 0.1, "LLE {lle}, PCA {pca}");

    let n_labels = labels.iter().max().unwrap() + 1;
    let centre = |l: usize| {
        let rows: Vec<usize> = (0..y.nrows()).filter(|&i| labels[i] == l).collect();
        rows.iter()
            .fold(nalgebra::RowDVector::zeros(2), |s, &i| s + y.row(i))
            / rows.len() as f32
    };
    let centres: Vec<_> = (0..n_labels).map(centre).collect();
    // None squashed to a point.
    for (l, c) in centres.iter().enumerate() {
        let spread = (0..y.nrows())
            .filter(|&i| labels[i] == l)
            .map(|i| (y.row(i) - c).norm())
            .fold(0.0_f32, f32::max);
        assert!(spread > 0.0, "cluster {l} collapsed");
    }
    // And not on top of one another.
    let astray = (0..y.nrows())
        .filter(|&i| {
            let d = |l: usize| (y.row(i) - &centres[l]).norm();
            (0..n_labels).any(|l| d(l) < d(labels[i]))
        })
        .count();
    assert!(
        astray * 20 <= y.nrows(),
        "{astray} points nearer another cluster"
    );
}

#[test]
fn points_with_few_links_are_placed_by_them() {
    // A sheet, plus a point linked to only two points of it and one linked
    // to none: neither can anchor LLE's weights, both still get a place.
    let (data, _) = swiss_roll(300, 4);
    let mut edges = fuzzy_edges(&data, 10);
    let n = data.nrows();
    let data = data.resize_vertically(n + 2, 0.0);
    edges.push((7, n, 1.0));
    edges.push((8, n, 1.0));
    let y = lle_layout_2d_on_graph(&data, &edges);
    assert_eq!(y.shape(), (n + 2, 2));
    assert!(y.iter().all(|v| v.is_finite()), "{y}");
    // The sheet is laid out (a node it cannot weigh must not squash it),
    // and the two-link point sits between its two.
    assert!((y.row(7) - y.row(8)).norm() > 0.0, "the sheet collapsed");
    let between = (y.row(7) + y.row(8)) / 2.0;
    assert!(
        (y.row(n) - &between).norm() < 1e-4,
        "{} vs {between}",
        y.row(n)
    );
}

#[test]
fn tiny_inputs_get_a_layout_of_their_size() {
    for n in [0, 1, 2, 3, 5, 12] {
        let data = Mat::from_fn(n, 3, |i, c| ((i * 3 + c) as f32).sin());
        let edges: Vec<(usize, usize, f32)> = (1..n).map(|i| (i - 1, i, 1.0)).collect();
        let y = lle_layout_2d_on_graph(&data, &edges);
        assert_eq!(y.shape(), (n, 2));
        assert!(y.iter().all(|v| v.is_finite()), "n {n}: {y}");
    }
}
