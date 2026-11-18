//! Lay out a table too large for a layout by coarsening it: its columns'
//! fuzzy kNN graph is coarsened into pseudobulks (PBs) by heavy-edge
//! matching ([`crate::matrix::coarsen`]), the graph collapsed onto the PBs,
//! any `*_on_graph` layout run on that PB graph, and every column placed
//! from the PBs near it in that graph ([`Coarsened::place_cells`]).
//!
//! ```ignore
//! let pbs = coarsened(&table_kn, &CoarsenArgs::default())?;
//! let pb_coords = stress_layout_2d_on_graph(pbs.n_pb(), &pbs.pb_edges, 42);
//! let cell_coords = pbs.place_cells(&table_kn, &pb_coords, 3, 10.0);
//! ```

#[cfg(doc)]
use super::project_cells_nystrom;
use super::Mat;
use crate::matrix::coarsen::{coarsen, collapse};
use crate::matrix::knn_graph::{KnnGraph, KnnGraphArgs};
use rayon::prelude::*;

type Edges = Vec<(usize, usize, f32)>;

/// Args of [`coarsened`].
#[derive(Debug, Clone)]
pub struct CoarsenArgs {
    /// Neighbours per column in the kNN graph, counting the column itself
    /// (uwot's `n_neighbors`).
    pub knn: usize,
    /// How many PBs to coarsen into (fewer if the graph has fewer nodes).
    pub n_landmarks: usize,
    /// Block size of the kNN search.
    pub block_size: usize,
    /// Seed of the matching order.
    pub seed: u64,
}

impl Default for CoarsenArgs {
    fn default() -> Self {
        Self {
            knn: 15,
            n_landmarks: 1000,
            block_size: 1000,
            seed: 42,
        }
    }
}

/// A table's columns grouped into PBs, and the graph between the PBs.
#[derive(Debug, Clone)]
pub struct Coarsened {
    /// Each column's PB, `usize::MAX` for a column in none.
    pub membership: Vec<usize>,
    /// The column graph collapsed onto the PBs: summed weights, `a < b`, no
    /// self loops.
    pub pb_edges: Edges,
    /// Each PB's mean column, `dims × n_pb`.
    pub pb_features: Mat,
    /// Columns per PB.
    pub pb_size: Vec<usize>,
}

/// The undirected fuzzy kNN graph of the columns of `table_kn`
/// (`dims × n`), as uwot builds it with `knn` counting each column itself:
/// `(i, j, membership)` with `i < j`.
pub fn fuzzy_graph_of_columns(
    table_kn: &Mat,
    knn: usize,
    block_size: usize,
) -> anyhow::Result<Edges> {
    KnnGraph::fuzzy_edges_of_rows(
        &table_kn.transpose(),
        KnnGraphArgs {
            knn: knn.saturating_sub(1).max(1),
            block_size,
            reciprocal: false,
        },
    )
}

/// The columns of `table_kn` (`dims × n`, one column per cell) coarsened
/// into about `args.n_landmarks` PBs on their fuzzy kNN graph.
pub fn coarsened(table_kn: &Mat, args: &CoarsenArgs) -> anyhow::Result<Coarsened> {
    let n = table_kn.ncols();
    let edges = fuzzy_graph_of_columns(table_kn, args.knn, args.block_size)?;
    let membership = coarsen(n, &edges, args.n_landmarks.clamp(1, n.max(1)), args.seed);
    Ok(Coarsened::from_membership(table_kn, &edges, membership))
}

impl Coarsened {
    /// PBs from a given grouping of the columns of `table_kn` (ids `0..`,
    /// `usize::MAX` for none) and the column graph `edges`.
    #[must_use]
    pub fn from_membership(
        table_kn: &Mat,
        edges: &[(usize, usize, f32)],
        membership: Vec<usize>,
    ) -> Self {
        let n_pb = membership
            .iter()
            .filter(|&&g| g != usize::MAX)
            .max()
            .map_or(0, |&g| g + 1);
        let mut pb_size = vec![0_usize; n_pb];
        let mut pb_features = Mat::zeros(table_kn.nrows(), n_pb);
        for (c, &g) in membership.iter().enumerate() {
            if g != usize::MAX {
                pb_size[g] += 1;
                let mut col = pb_features.column_mut(g);
                col += table_kn.column(c);
            }
        }
        for (g, &s) in pb_size.iter().enumerate() {
            if s > 0 {
                pb_features.column_mut(g).unscale_mut(s as f32);
            }
        }
        let pb_edges = collapse(edges, &membership);
        Self {
            membership,
            pb_edges,
            pb_features,
            pb_size,
        }
    }

    /// Number of PBs.
    #[must_use]
    pub fn n_pb(&self) -> usize {
        self.pb_size.len()
    }

    /// Every column of `table_kn` placed on the PB layout `pb_coords`
    /// (`n_pb × 2`) at the alpha-decay-kernel mean of PBs near it, as
    /// [`project_cells_nystrom`] weighs them: `σ` is the distance to the
    /// `knn`-th nearest candidate PB and `alpha` the kernel exponent.
    ///
    /// A column's candidates are its own PB and that PB's neighbours in
    /// `pb_edges`, so it never lands between PBs the graph keeps apart
    /// (averaging two distant groups puts a cell in the empty space
    /// between them, a streak across the layout). A column in no PB has
    /// every PB as a candidate. Returns `n × 2`.
    #[must_use]
    pub fn place_cells(&self, table_kn: &Mat, pb_coords: &Mat, knn: usize, alpha: f32) -> Mat {
        let n_pb = self.n_pb();
        let mut near: Vec<Vec<usize>> = (0..n_pb).map(|g| vec![g]).collect();
        for &(a, b, _) in &self.pb_edges {
            near[a].push(b);
            near[b].push(a);
        }
        let every: Vec<usize> = (0..n_pb).collect();
        let placed: Vec<(f32, f32)> = (0..table_kn.ncols())
            .into_par_iter()
            .map(|c| {
                let pbs = match self.membership.get(c) {
                    Some(&g) if g < n_pb => &near[g],
                    _ => &every,
                };
                let cell = table_kn.column(c);
                let dist: Vec<f32> = pbs
                    .iter()
                    .map(|&g| (cell - self.pb_features.column(g)).norm())
                    .collect();
                let w = alpha_decay(&dist, knn, alpha);
                pbs.iter().zip(&w).fold((0.0, 0.0), |(x, y), (&g, &w)| {
                    (x + w * pb_coords[(g, 0)], y + w * pb_coords[(g, 1)])
                })
            })
            .collect();
        let mut coords = Mat::zeros(placed.len(), 2);
        for (c, (x, y)) in placed.into_iter().enumerate() {
            coords[(c, 0)] = x;
            coords[(c, 1)] = y;
        }
        coords
    }
}

/// Weights summing to 1 from distances `dist`: `exp(-(d/σ)^alpha)` with
/// `σ` the `knn`-th smallest distance; all on the nearest when every
/// weight underflows.
fn alpha_decay(dist: &[f32], knn: usize, alpha: f32) -> Vec<f32> {
    if dist.is_empty() {
        return Vec::new();
    }
    let kth = knn.clamp(1, dist.len()) - 1;
    let mut sorted = dist.to_vec();
    sorted.select_nth_unstable_by(kth, f32::total_cmp);
    let sigma = sorted[kth].max(1e-6);
    let mut w: Vec<f32> = dist
        .iter()
        .map(|&d| (-(d / sigma).powf(alpha)).exp())
        .collect();
    let sum: f32 = w.iter().sum();
    if sum > 1e-12 {
        w.iter_mut().for_each(|v| *v /= sum);
    } else {
        let nearest = (0..dist.len())
            .min_by(|&a, &b| dist[a].total_cmp(&dist[b]))
            .unwrap_or(0);
        w.fill(0.0);
        w[nearest] = 1.0;
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::layout::{stress_layout_2d_on_graph, umap_layout_2d_on_graph};
    use crate::matrix::umap::Umap;
    use rand::{rngs::SmallRng, RngExt, SeedableRng};

    const PER: usize = 60;

    /// Three blobs of `PER` cells in 4 dims, uniform jitter around their
    /// centres.
    fn blobs() -> Mat {
        let mut rng = SmallRng::seed_from_u64(7);
        Mat::from_fn(4, 3 * PER, |r, c| {
            let centre = if r == c / PER { 20.0 } else { 0.0 };
            centre + rng.random_range(-1.0..1.0)
        })
    }

    /// One coordinate pair per cell, and cells nearer their own PB, on
    /// average, than PBs are to each other.
    fn check(name: &str, table: &Mat, pbs: &Coarsened, pb_coords: &Mat) {
        let cells = pbs.place_cells(table, pb_coords, 2, 10.0);
        assert_eq!((cells.nrows(), cells.ncols()), (table.ncols(), 2));
        assert!(cells.iter().all(|v| v.is_finite()));
        let n = cells.nrows();
        let dist = |c: usize, h: usize| (cells.row(c) - pb_coords.row(h)).norm();
        let own = (0..n).map(|c| dist(c, pbs.membership[c])).sum::<f32>() / n as f32;
        let any = (0..n)
            .flat_map(|c| (0..pbs.n_pb()).map(move |h| (c, h)))
            .map(|(c, h)| dist(c, h))
            .sum::<f32>()
            / (n * pbs.n_pb()) as f32;
        assert!(
            own < 0.5 * any,
            "{name}: {own} from their own PB, {any} from any"
        );
    }

    #[test]
    fn coarsened_cells_land_near_their_pb() {
        let table = blobs();
        let args = CoarsenArgs {
            knn: 10,
            n_landmarks: 12,
            ..CoarsenArgs::default()
        };
        let pbs = coarsened(&table, &args).unwrap();
        assert!(pbs.n_pb() >= 3 && pbs.n_pb() < table.ncols());
        assert_eq!(pbs.pb_size.iter().sum::<usize>(), table.ncols());
        assert_eq!(pbs.pb_features.ncols(), pbs.n_pb());
        assert!(pbs.pb_edges.iter().all(|&(a, b, w)| a < b && w > 0.0));

        check(
            "stress",
            &table,
            &pbs,
            &stress_layout_2d_on_graph(pbs.n_pb(), &pbs.pb_edges, 1),
        );
        check(
            "umap",
            &table,
            &pbs,
            &umap_layout_2d_on_graph(pbs.n_pb(), &pbs.pb_edges, &Umap::default()),
        );
    }

    #[test]
    fn a_cell_is_not_placed_between_unlinked_pbs() {
        // Two PBs with no edge between them; cell 4 sits halfway in
        // feature space but belongs to PB 0.
        let table = Mat::from_row_slice(1, 5, &[0.0, 0.1, 1.0, 1.1, 0.55]);
        let edges = [(0, 1, 1.0), (2, 3, 1.0), (0, 4, 1.0)];
        let pbs = Coarsened::from_membership(&table, &edges, vec![0, 0, 1, 1, 0]);
        assert!(pbs.pb_edges.is_empty());
        let pb_coords = Mat::from_row_slice(2, 2, &[0.0, 0.0, 10.0, 0.0]);
        let cells = pbs.place_cells(&table, &pb_coords, 2, 10.0);
        assert!(cells[(4, 0)].abs() < 1e-4, "{}", cells[(4, 0)]);
        assert!((cells[(2, 0)] - 10.0).abs() < 1e-4);
        let unplaced = Coarsened::from_membership(&table, &edges, vec![0, 0, 1, 1, usize::MAX]);
        let between = unplaced.place_cells(&table, &pb_coords, 2, 10.0)[(4, 0)];
        assert!(between > 1.0 && between < 9.0, "{between}");
    }
}
