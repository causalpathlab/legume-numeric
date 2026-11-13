use super::{correct, harmonize, HarmonyArgs};
use nalgebra::DMatrix;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rand_distr::StandardNormal;

/// Each batch's mean cell, dims × batches.
fn batch_means(m: &DMatrix<f64>, batch: &[usize], n_batches: usize) -> DMatrix<f64> {
    let mut sums = DMatrix::<f64>::zeros(m.nrows(), n_batches);
    let mut counts = vec![0.0; n_batches];
    for (i, &b) in batch.iter().enumerate() {
        let mut col = sums.column_mut(b);
        col += m.column(i);
        counts[b] += 1.0;
    }
    for (b, &c) in counts.iter().enumerate() {
        sums.column_mut(b).unscale_mut(c);
    }
    sums
}

/// One cluster holding every cell, two batches of `n` cells: the ridge
/// solution (paper section 3, intercept unpenalised) moves each batch by
/// `n (m_b - m) / (n + lambda)`, so the corrected batch means differ by
/// exactly `lambda / (n + lambda)` of the original difference, the overall
/// mean (the unpenalised intercept) stays, and each cell keeps its offset
/// from its batch mean.
#[test]
fn ridge_correction_shrinks_the_batch_difference_exactly() {
    let n = 50;
    let lambda = 1.0;
    let mut rng = SmallRng::seed_from_u64(7);
    let batch: Vec<usize> = (0..2 * n).map(|i| i / n).collect();
    let z = DMatrix::from_fn(3, 2 * n, |c, i| {
        let shift = if batch[i] == 1 {
            [4.0, -2.0, 0.5][c]
        } else {
            0.0
        };
        shift + rng.sample::<f64, _>(StandardNormal)
    });
    let r = DMatrix::from_element(1, 2 * n, 1.0);
    let cells: Vec<Vec<usize>> = (0..2).map(|b| (b * n..(b + 1) * n).collect()).collect();
    let zc = correct(&z, &z.transpose(), &r, &cells, lambda);
    let (m, mc) = (batch_means(&z, &batch, 2), batch_means(&zc, &batch, 2));
    for c in 0..3 {
        let before = m[(c, 1)] - m[(c, 0)];
        let after = mc[(c, 1)] - mc[(c, 0)];
        let want = lambda / (n as f64 + lambda) * before;
        assert!((after - want).abs() < 1e-9, "dim {c}: {after} vs {want}");
        let overall = |m: &DMatrix<f64>| (m[(c, 0)] + m[(c, 1)]) / 2.0;
        assert!(
            (overall(&mc) - overall(&m)).abs() < 1e-9,
            "dim {c}: mean moved"
        );
        for (i, &b) in batch.iter().enumerate() {
            let off = (z[(c, i)] - m[(c, b)]) - (zc[(c, i)] - mc[(c, b)]);
            assert!(off.abs() < 1e-9);
        }
    }
}

/// Of each cell's `k` nearest neighbours (Euclidean, rows of `x`), the
/// mean share in another batch and the mean share of its own type.
fn knn_shares(x: &DMatrix<f32>, batch: &[usize], kind: &[usize], k: usize) -> (f64, f64) {
    let n = x.nrows();
    let (mut other, mut same) = (0usize, 0usize);
    for i in 0..n {
        let mut d: Vec<(f32, usize)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| ((x.row(i) - x.row(j)).norm_squared(), j))
            .collect();
        d.sort_by(|a, b| a.0.total_cmp(&b.0));
        for &(_, j) in &d[..k] {
            other += usize::from(batch[j] != batch[i]);
            same += usize::from(kind[j] == kind[i]);
        }
    }
    let t = (n * k) as f64;
    (other as f64 / t, same as f64 / t)
}

/// Two cell types split along one PC, and a second batch shifted along
/// another by as much as the types are apart, so far that soft k-means
/// without the diversity penalty keeps the batches in separate clusters
/// (and the ridge correction then has nothing to remove). Before Harmony a
/// cell's neighbours are all from its own batch; after it, at the paper's
/// defaults, at least a third are from the other batch (half is perfect
/// mixing), while at least 95% stay of the cell's own type.
#[test]
fn harmony_mixes_batches_and_keeps_types_apart() {
    let (n, d) = (600, 10);
    let mut rng = SmallRng::seed_from_u64(11);
    let batch: Vec<usize> = (0..n).map(|i| i % 2).collect();
    let kind: Vec<usize> = (0..n).map(|i| (i / 2) % 2).collect();
    let x = DMatrix::from_fn(n, d, |i, c| {
        let mut v = 0.5 * rng.sample::<f32, _>(StandardNormal);
        if c == 0 {
            v += if kind[i] == 0 { 3.0 } else { -3.0 };
        }
        if c == 1 && batch[i] == 1 {
            v += 6.0;
        }
        v
    });
    let (other, same) = knn_shares(&x, &batch, &kind, 15);
    assert!(other < 0.05, "before: other-batch share {other}");
    assert!(same > 0.99, "before: own-type share {same}");
    let h = harmonize(&x, &batch, &HarmonyArgs::default(), 1).unwrap();
    assert_eq!(h.shape(), x.shape());
    let (other, same) = knn_shares(&h, &batch, &kind, 15);
    eprintln!("after Harmony: other-batch {other:.3}, own-type {same:.3}");
    assert!(other > 1.0 / 3.0, "after: other-batch share {other}");
    assert!(same > 0.95, "after: own-type share {same}");
}

/// Settings that cannot correct are refused: no ridge penalty makes every
/// cluster's system singular, no width makes every assignment NaN, and
/// zero clusters or more clusters than cells cannot be run as asked. So
/// is a non-finite input, which would turn every output NaN.
#[test]
fn harmony_refuses_settings_that_cannot_correct() {
    let x = DMatrix::from_fn(40, 3, |i, j| (i * 3 + j) as f32 * 0.1);
    let batch: Vec<usize> = (0..40).map(|i| i % 2).collect();
    for p in [
        HarmonyArgs {
            lambda: 0.0,
            ..Default::default()
        },
        HarmonyArgs {
            sigma: 0.0,
            ..Default::default()
        },
        HarmonyArgs {
            theta: -1.0,
            ..Default::default()
        },
        HarmonyArgs {
            max_rounds: 0,
            ..Default::default()
        },
        HarmonyArgs {
            n_clusters: Some(0),
            ..Default::default()
        },
        HarmonyArgs {
            n_clusters: Some(41),
            ..Default::default()
        },
    ] {
        assert!(harmonize(&x, &batch, &p, 1).is_err(), "{p:?}");
    }
    for bad in [f32::NAN, f32::INFINITY] {
        let mut y = x.clone();
        y[(5, 1)] = bad;
        assert!(
            harmonize(&y, &batch, &HarmonyArgs::default(), 1).is_err(),
            "{bad}"
        );
    }
}
