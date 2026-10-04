//! Neighbour lists of every row of a matrix among its other rows, exact or
//! by the inverted-file search depending on size.

use crate::matrix::knn::all_pairs::knn_rows_l2;
use crate::matrix::knn::ivf::{knn_rows_ivf, IvfArgs, DEFAULT_N_PROBE};
use crate::matrix::knn::KNN_SEED;
use log::info;
use nalgebra::DMatrix;

/// Up to this many points every row's neighbours come from the exact
/// all-pairs Gram kernel; beyond it from the inverted-file search. Both are
/// parallel and thread-count independent; the split is where `O(n²)` stops
/// being affordable.
pub const ALL_PAIRS_THRESHOLD: usize = 65_536;

/// The nearest neighbours of every row of `rows` among the other rows, as
/// `(indices, Euclidean distances)` per row, nearest first. A row is never
/// its own neighbour.
///
/// Up to [`ALL_PAIRS_THRESHOLD`] rows the search is exact (all-pairs Gram
/// kernel): each row gets `min(n_neighbours, n − 1)` neighbours, and a row with
/// a non-finite entry sorts last. Beyond it the inverted-file search is
/// approximate, seeded and probes [`DEFAULT_N_PROBE`] cells: a row can come
/// back with fewer than `n_neighbours` when the probed cells hold fewer
/// candidates, and rows are expected to be finite and to have at least one
/// column.
pub fn knn_rows(rows: &DMatrix<f32>, n_neighbours: usize) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
    let nn = rows.nrows();
    if nn <= ALL_PAIRS_THRESHOLD {
        info!("kNN by the exact all-pairs kernel over {nn} points");
        knn_rows_l2(rows, n_neighbours)
    } else {
        info!("kNN by the inverted-file search over {nn} points");
        knn_rows_ivf(
            rows,
            &IvfArgs {
                k: n_neighbours,
                n_lists: 0,
                n_probe: DEFAULT_N_PROBE,
                seed: KNN_SEED,
            },
        )
    }
}
