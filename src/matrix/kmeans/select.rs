//! Several seeded starts: the best of a few [`kmeans_rows_seeded`] fits by
//! their within-cluster sum of squares, since one start can stop in a poor
//! local optimum. To choose the number of clusters, see
//! [`crate::matrix::gmm::choose_gmm`], whose soft likelihood does not
//! reward cutting a Gaussian in two as a BIC on hard assignments does.

use super::{kmeans_rows_seeded, KmeansRowsFit, KmeansRowsOpts};
use nalgebra::DMatrix;
use rayon::prelude::*;

/// The sum over rows of `z` of the squared Euclidean distance to the
/// centroid each is labelled with.
#[must_use]
pub fn kmeans_rss(z: &DMatrix<f32>, fit: &KmeansRowsFit) -> f64 {
    (0..z.nrows())
        .into_par_iter()
        .map(|i| {
            let c = fit.labels[i];
            (0..z.ncols())
                .map(|j| f64::from(z[(i, j)] - fit.centroids[(c, j)]).powi(2))
                .sum::<f64>()
        })
        .sum()
}

/// The fit of least [`kmeans_rss`] among `n_starts` (at least one) runs of
/// [`kmeans_rows_seeded`], the `s`-th seeded `opts.seed + s`; a tie goes to
/// the earliest start.
#[must_use]
pub fn kmeans_rows_best(z: &DMatrix<f32>, opts: &KmeansRowsOpts, n_starts: usize) -> KmeansRowsFit {
    let mut best: Option<(f64, KmeansRowsFit)> = None;
    for s in 0..n_starts.max(1) as u64 {
        let fit = kmeans_rows_seeded(
            z,
            &KmeansRowsOpts {
                seed: opts.seed.wrapping_add(s),
                ..opts.clone()
            },
        );
        let rss = kmeans_rss(z, &fit);
        if best.as_ref().is_none_or(|(least, _)| rss < *least) {
            best = Some((rss, fit));
        }
    }
    best.expect("at least one start").1
}
