//! UMAP of a table's columns, every point laid out (no coarsening): their
//! fuzzy kNN graph, a centred PCA(2) start ([`centred_pc_init_2d`]), then
//! [`Umap::fit`], as [`UmapLayoutArgs`] sets them.
//!
//! [`joint_umap_layout_2d_of_columns`] lays out two point sets that share
//! one space (cells and the features co-embedded with them) in one map.

use super::coarsened::{fuzzy_graph_of_rows, Edges};
use super::{fit_umap_2d, Mat};
use crate::matrix::knn::{ColumnDict, SearchScratch};
use crate::matrix::knn_graph::{fuzzy_union, umap_memberships};
use crate::matrix::pca::centred_pc_init_2d;
use crate::matrix::umap::Umap;
use log::info;
use rayon::prelude::*;

/// Args of [`umap_layout_2d_of_columns`] and
/// [`joint_umap_layout_2d_of_columns`]. The default is uwot's t-UMAP.
#[derive(Debug, Clone)]
pub struct UmapLayoutArgs {
    /// Neighbours per point in the kNN graph, counting the point itself
    /// (uwot's `n_neighbors`).
    pub knn: usize,
    /// Block size of the kNN search.
    pub block_size: usize,
    /// SGD epochs; `None` for [`Umap::default_epochs`] of the points laid
    /// out. Overrides `sgd.n_epochs`.
    pub epochs: Option<usize>,
    /// The SGD (kernel, learning rate, negative sampling, seed).
    pub sgd: Umap,
}

impl Default for UmapLayoutArgs {
    fn default() -> Self {
        Self {
            knn: 15,
            block_size: 1000,
            epochs: None,
            sgd: Umap::tumap(),
        }
    }
}

impl UmapLayoutArgs {
    /// The SGD for `n` points, its epochs resolved.
    #[must_use]
    pub fn umap(&self, n: usize) -> Umap {
        Umap {
            n_epochs: self.epochs.unwrap_or_else(|| Umap::default_epochs(n)),
            ..self.sgd.clone()
        }
    }
}

/// UMAP of the columns of `table_kn` (`dims × n`). Returns `n × 2`.
pub fn umap_layout_2d_of_columns(table_kn: &Mat, args: &UmapLayoutArgs) -> anyhow::Result<Mat> {
    let rows = table_kn.transpose();
    let edges = fuzzy_graph_of_rows(&rows, args.knn, args.block_size)?;
    Ok(fit_from_pca(rows, &edges, args))
}

/// UMAP of cells and features that share one space: `cells_kn`
/// (`dims × cells`) and `features_kn` (`dims × features`, e.g. a
/// co-embedding) become the points of one graph. It is the fuzzy kNN graph
/// over all of them, plus each feature's `args.knn` nearest cells, so no
/// feature is tied only to other features and left on an island beside its
/// cells. One SGD from a joint PCA(2) start. Returns (`cells × 2`,
/// `features × 2`).
pub fn joint_umap_layout_2d_of_columns(
    cells_kn: &Mat,
    features_kn: &Mat,
    args: &UmapLayoutArgs,
) -> anyhow::Result<(Mat, Mat)> {
    let (h, n, d) = (cells_kn.nrows(), cells_kn.ncols(), features_kn.ncols());
    anyhow::ensure!(n > 0, "no cells to lay the features out among");
    anyhow::ensure!(
        features_kn.nrows() == h,
        "features have {} dims but cells {h}: not one shared space",
        features_kn.nrows()
    );
    // Every point as a row: cells, then features.
    let mut rows = Mat::zeros(n + d, h);
    rows.rows_mut(0, n).tr_copy_from(cells_kn);
    rows.rows_mut(n, d).tr_copy_from(features_kn);

    let mut edges = fuzzy_graph_of_rows(&rows, args.knn, args.block_size)?;
    edges.extend(feature_to_cell_edges(cells_kn, features_kn, args.knn)?);
    let edges = fuzzy_union(edges);
    info!(
        "Joint graph: {n} cells + {d} features, {} edges",
        edges.len()
    );
    let coords = fit_from_pca(rows, &edges, args);
    Ok((
        coords.rows(0, n).into_owned(),
        coords.rows(n, d).into_owned(),
    ))
}

/// The SGD of the points `rows` (`n × dims`, consumed) on `edges`, from
/// their centred PCA(2). Returns `n × 2`.
fn fit_from_pca(rows: Mat, edges: &Edges, args: &UmapLayoutArgs) -> Mat {
    let umap = args.umap(rows.nrows());
    let init = centred_pc_init_2d(rows, umap.seed);
    info!(
        "UMAP SGD of {} points (epochs={}) ...",
        init.len() / 2,
        umap.n_epochs
    );
    fit_umap_2d(&umap, edges, &init)
}

/// Each feature to its nearest cells in the shared space (`knn` counting
/// the feature itself, as uwot counts `n_neighbors`; at most as many as
/// the cells' index returns), weighted by UMAP's membership of that
/// feature in them ([`umap_memberships`]). Edges are
/// `(cell, n + feature, w)`.
fn feature_to_cell_edges(cells_kn: &Mat, features_kn: &Mat, knn: usize) -> anyhow::Result<Edges> {
    let (h, n) = cells_kn.shape();
    // Views onto the cells, not a copy of the table.
    let index = ColumnDict::from_dvector_views(cells_kn.column_iter().collect(), (0..n).collect());
    let k = knn.saturating_sub(1).max(1);
    let per_feature: Vec<Edges> = features_kn
        .as_slice()
        .par_chunks(h)
        .enumerate()
        .map_init(SearchScratch::default, |scratch, (f, query)| {
            let (cells, dists) = index.search_by_query_data_reuse(query, k, scratch)?;
            Ok(cells
                .into_iter()
                .zip(umap_memberships(&dists))
                .map(|(c, w)| (c, n + f, w))
                .collect())
        })
        .collect::<anyhow::Result<_>>()?;
    Ok(per_feature.into_iter().flatten().collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    /// Two groups of cells in a shared 6-D space, and features co-embedded
    /// at each group's centre (feature `f` with group `f % 2`). Features of
    /// a group are nearly identical, so a kNN over the union alone would
    /// tie them mostly to each other.
    fn two_groups(per: usize, features: usize) -> (Mat, Mat) {
        let h = 6;
        let mut rng = SmallRng::seed_from_u64(7);
        let centre = |g: usize, d: usize| if d % 2 == g { 4.0 } else { 0.0 };
        let cells = Mat::from_fn(h, 2 * per, |d, c| {
            centre(c / per, d) + rng.random_range(-1.0..1.0)
        });
        let feats = Mat::from_fn(h, features, |d, f| {
            centre(f % 2, d) + rng.random_range(-0.05..0.05)
        });
        (cells, feats)
    }

    fn args(seed: u64) -> UmapLayoutArgs {
        UmapLayoutArgs {
            block_size: 128,
            epochs: Some(200),
            sgd: Umap {
                seed,
                ..Umap::tumap()
            },
            ..UmapLayoutArgs::default()
        }
    }

    type Xy = [f32; 2];

    fn xy(m: &Mat, i: usize) -> Xy {
        [m[(i, 0)], m[(i, 1)]]
    }

    fn dist(p: Xy, q: Xy) -> f32 {
        ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt()
    }

    fn centroid(m: &Mat, rows: std::ops::Range<usize>) -> Xy {
        let k = rows.len() as f32;
        rows.fold([0.0, 0.0], |[x, y], r| {
            [x + m[(r, 0)] / k, y + m[(r, 1)] / k]
        })
    }

    /// The median distance of `rows` to their centroid.
    fn radius(m: &Mat, rows: std::ops::Range<usize>) -> f32 {
        let c = centroid(m, rows.clone());
        let mut d: Vec<f32> = rows.map(|r| dist(xy(m, r), c)).collect();
        d.sort_by(f32::total_cmp);
        d[d.len() / 2]
    }

    #[test]
    fn a_larger_min_dist_spreads_a_group_out() {
        let per = 200;
        let (cells, _) = two_groups(per, 0);
        // A group's radius over the distance between the two centroids.
        let looseness_at = |min_dist: f32, seed: u64| {
            let a = UmapLayoutArgs {
                sgd: args(seed).sgd.with_min_dist(min_dist, 1.0).unwrap(),
                ..args(seed)
            };
            let m = umap_layout_2d_of_columns(&cells, &a).unwrap();
            assert_eq!(m.shape(), (2 * per, 2));
            radius(&m, 0..per) / dist(centroid(&m, 0..per), centroid(&m, per..2 * per))
        };
        // The parallel SGD is not deterministic, so one run's ratio varies
        // widely; the mean over a few seeds is stable.
        let looseness =
            |min_dist: f32| (1..=3).map(|s| looseness_at(min_dist, s)).sum::<f32>() / 3.0;
        let (tight, loose) = (looseness(0.0), looseness(0.9));
        assert!(
            loose > 1.5 * tight,
            "min_dist 0.9 gives {loose:.3}, min_dist 0 gives {tight:.3}"
        );
    }

    #[test]
    fn a_joint_layout_puts_each_feature_among_its_own_cells() {
        let per = 200;
        let (cells, feats) = two_groups(per, 40);
        let (cell_xy, feat_xy) = joint_umap_layout_2d_of_columns(&cells, &feats, &args(1)).unwrap();
        assert_eq!(cell_xy.shape(), (2 * per, 2));
        assert_eq!(feat_xy.shape(), (40, 2));
        let group = |g: usize| centroid(&cell_xy, g * per..(g + 1) * per);
        for f in 0..40 {
            let at = xy(&feat_xy, f);
            assert!(
                dist(at, group(f % 2)) < dist(at, group(1 - f % 2)),
                "feature {f} sits nearer the other group's cells"
            );
        }
    }

    #[test]
    fn features_sit_inside_their_cells_not_on_an_island_beside_them() {
        // A small population with more features than cells: a kNN over the
        // union alone ties each feature to other features, and they drift
        // off as a clump beside their cells.
        let (per, n_feats) = (200, 800);
        let (cells, feats) = two_groups(per, n_feats);
        let (cell_xy, feat_xy) = joint_umap_layout_2d_of_columns(&cells, &feats, &args(1)).unwrap();
        for f in 0..n_feats {
            let at = xy(&feat_xy, f);
            let (nearest, d) = (0..2 * per)
                .map(|c| (c, dist(at, xy(&cell_xy, c))))
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap();
            let g = f % 2;
            assert_eq!(
                nearest / per,
                g,
                "feature {f}'s nearest cell is of the other group"
            );
            // Features take room on the map too, so allow a margin past the
            // cells' median radius (1.5× missed by a hair once, as the SGD
            // order varies); a clump beside the cells lies well beyond.
            let r = radius(&cell_xy, g * per..(g + 1) * per);
            assert!(
                d < 2.0 * r,
                "feature {f} is {d:.2} from the nearest cell; radius {r:.2}"
            );
        }
    }

    #[test]
    fn features_of_another_space_are_refused() {
        let (cells, _) = two_groups(10, 0);
        let feats = Mat::zeros(5, 3);
        assert!(joint_umap_layout_2d_of_columns(&cells, &feats, &args(1)).is_err());
    }

    #[test]
    fn no_cells_is_refused() {
        let feats = Mat::zeros(6, 3);
        let err = joint_umap_layout_2d_of_columns(&Mat::zeros(6, 0), &feats, &args(1));
        assert!(err.unwrap_err().to_string().contains("no cells"));
    }

    /// Past the exact scan's size the cells' index is an HNSW, which returns
    /// at most its beam width (128) of neighbours, debug build or not.
    #[test]
    fn features_are_tied_to_no_more_cells_than_the_index_returns() {
        let n = 8_200;
        let mut rng = SmallRng::seed_from_u64(3);
        let cells = Mat::from_fn(2, n, |_, _| rng.random_range(0.0..1.0));
        let feats = Mat::from_fn(2, 2, |_, _| rng.random_range(0.0..1.0));
        let edges = feature_to_cell_edges(&cells, &feats, 300).unwrap();
        for f in 0..2 {
            assert_eq!(edges.iter().filter(|e| e.1 == n + f).count(), 128);
        }
    }
}
