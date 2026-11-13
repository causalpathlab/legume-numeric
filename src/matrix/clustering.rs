//! K-means clustering traits for matrices
//!
//! Provides traits for clustering rows or columns of matrices using the
//! `clustering` crate, plus a Leiden community-detection helper that builds
//! a kNN graph in latent space and runs modularity-objective Leiden.

use crate::matrix::knn_graph::{self, KnnGraph, KnnGraphArgs};
use crate::matrix::traits::MatOps;
use log::info;
use nalgebra::DMatrix;
use std::borrow::Cow;

/// Arguments for k-means clustering
#[derive(Debug, Clone)]
pub struct KmeansArgs {
    /// Number of clusters
    pub num_clusters: usize,
    /// Maximum number of iterations
    pub max_iter: usize,
}

impl Default for KmeansArgs {
    fn default() -> Self {
        Self {
            num_clusters: 1,
            max_iter: 100,
        }
    }
}

impl KmeansArgs {
    /// Create args with specified number of clusters
    pub fn with_clusters(num_clusters: usize) -> Self {
        Self {
            num_clusters,
            ..Default::default()
        }
    }
}

/// Trait for k-means clustering on matrices
pub trait Kmeans {
    /// Cluster columns and return membership vector
    ///
    /// # Arguments
    /// * `args` - Clustering parameters
    ///
    /// # Returns
    /// Vector of cluster assignments, one per column
    fn kmeans_columns(&self, args: KmeansArgs) -> Vec<usize>;

    /// Cluster rows and return membership vector
    ///
    /// # Arguments
    /// * `args` - Clustering parameters
    ///
    /// # Returns
    /// Vector of cluster assignments, one per row
    fn kmeans_rows(&self, args: KmeansArgs) -> Vec<usize>;
}

impl<T> Kmeans for DMatrix<T>
where
    T: Clone + Sync + Send,
    Vec<T>: clustering::Elem,
{
    fn kmeans_columns(&self, args: KmeansArgs) -> Vec<usize> {
        if args.num_clusters <= 1 || self.ncols() == 0 {
            return vec![0; self.ncols()];
        }

        let data: Vec<Vec<T>> = self
            .column_iter()
            .map(|x| x.iter().cloned().collect())
            .collect();

        let clust = clustering::kmeans(args.num_clusters, &data, args.max_iter);
        clust.membership
    }

    fn kmeans_rows(&self, args: KmeansArgs) -> Vec<usize> {
        if args.num_clusters <= 1 || self.nrows() == 0 {
            return vec![0; self.nrows()];
        }

        let data: Vec<Vec<T>> = self
            .row_iter()
            .map(|x| x.iter().cloned().collect())
            .collect();

        let clust = clustering::kmeans(args.num_clusters, &data, args.max_iter);
        clust.membership
    }
}

/// How [`leiden_clustering_with`] prepares the rows before the kNN graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeidenInput {
    /// Row-L2-normalise, so kNN distances on the unit sphere are monotonic
    /// in cosine: for dot-product / cosine-trained embeddings (bge, gem).
    Cosine,
    /// Column z-score: Euclidean kNN on heterogeneous-scale dims (raw SVD).
    ZScore,
    /// The rows as given, e.g. PC scores whose scale already carries weight.
    AsIs,
}

/// Leiden community detection on a latent representation (rows = points).
///
/// Builds a kNN graph in latent space and runs the modularity-objective
/// Leiden algorithm. The community count is *not* a parameter: it emerges
/// from the modularity `resolution` (higher means more, smaller
/// communities), unless `target_clusters` is set, in which case the
/// resolution is binary-searched to approximate that count.
///
/// `cosine = true` is [`LeidenInput::Cosine`], `false` is
/// [`LeidenInput::ZScore`]; [`leiden_clustering_with`] also takes the rows
/// as they are.
///
/// Returns compacted (contiguous, 0-based) cluster labels, one per row.
pub fn leiden_clustering(
    latent: &DMatrix<f32>,
    knn: usize,
    resolution: f64,
    target_clusters: Option<usize>,
    seed: Option<u64>,
    cosine: bool,
) -> anyhow::Result<Vec<usize>> {
    let input = if cosine {
        LeidenInput::Cosine
    } else {
        LeidenInput::ZScore
    };
    leiden_clustering_with(latent, knn, resolution, target_clusters, seed, input)
}

/// [`leiden_clustering`] with the rows prepared by `input`.
pub fn leiden_clustering_with(
    latent: &DMatrix<f32>,
    knn: usize,
    resolution: f64,
    target_clusters: Option<usize>,
    seed: Option<u64>,
    input: LeidenInput,
) -> anyhow::Result<Vec<usize>> {
    let n = latent.nrows();
    let d = latent.ncols();
    if n < 2 {
        anyhow::bail!("Need at least 2 points for Leiden clustering");
    }
    info!("Leiden: {n} points x {d} features, knn={knn}, seed={seed:?}, input={input:?}");

    let latent_pre = if input == LeidenInput::AsIs {
        Cow::Borrowed(latent)
    } else {
        let mut x = latent.clone();
        if input == LeidenInput::Cosine {
            for mut row in x.row_iter_mut() {
                let denom = row.norm().max(1e-8);
                row /= denom;
            }
        } else {
            x.scale_columns_inplace();
        }
        Cow::Owned(x)
    };

    let (graph, weights) = KnnGraph::from_rows_fuzzy(
        &latent_pre,
        KnnGraphArgs {
            knn,
            block_size: 1000,
            reciprocal: false,
        },
    )?;
    info!(
        "KNN graph: {} nodes, {} edges",
        graph.num_nodes(),
        graph.num_edges()
    );

    let (network, total_edge_weight) = graph.to_leiden_network_with(&weights);
    let resolution_scaled = knn_graph::modularity_to_cpm_resolution(resolution, total_edge_weight);
    let seed_val = seed.map(|s| s as usize);

    let mut labels = if let Some(target_k) = target_clusters {
        knn_graph::tune_leiden_resolution(&network, n, target_k, resolution_scaled, seed_val)
    } else {
        knn_graph::run_leiden(&network, n, resolution_scaled, seed_val)
    };
    knn_graph::compact_labels(&mut labels);
    let n_clusters = labels.iter().copied().max().unwrap_or(0) + 1;
    info!("Leiden done: {n_clusters} clusters over {n} points");
    Ok(labels)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two groups apart by 10 along one dim, among 20 noise dims of sd 0.3:
    /// as is, every kNN edge stays in its group and Leiden at two clusters
    /// returns the groups (z-scoring would shrink the group dim to ±1
    /// against noise of sd 1). The bool form gives the same labels as its
    /// two variants.
    #[test]
    fn leiden_as_is_keeps_the_input_scale() {
        use rand::{rngs::SmallRng, RngExt, SeedableRng};
        use rand_distr::StandardNormal;
        let n = 200;
        let mut rng = SmallRng::seed_from_u64(3);
        let group: Vec<usize> = (0..n).map(|i| i % 2).collect();
        let x = DMatrix::from_fn(n, 21, |i, c| {
            let e: f32 = rng.sample(StandardNormal);
            if c == 0 {
                10.0 * group[i] as f32 + 0.1 * e
            } else {
                0.3 * e
            }
        });
        let matches = |labels: &[usize]| {
            let same = (0..n).filter(|&i| labels[i] == labels[0]).count();
            let agree = (0..n)
                .filter(|&i| (labels[i] == labels[0]) == (group[i] == group[0]))
                .count();
            same == n / 2 && agree == n
        };
        let run = |input| leiden_clustering_with(&x, 15, 1.0, Some(2), Some(1), input).unwrap();
        assert!(matches(&run(LeidenInput::AsIs)), "as is");
        assert_eq!(
            run(LeidenInput::ZScore),
            leiden_clustering(&x, 15, 1.0, Some(2), Some(1), false).unwrap()
        );
        assert_eq!(
            run(LeidenInput::Cosine),
            leiden_clustering(&x, 15, 1.0, Some(2), Some(1), true).unwrap()
        );
    }

    #[test]
    fn test_kmeans_columns_single_cluster() {
        let mat = DMatrix::from_row_slice(2, 4, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        let args = KmeansArgs::with_clusters(1);
        let membership = mat.kmeans_columns(args);

        assert_eq!(membership.len(), 4);
        assert!(membership.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_kmeans_columns_two_clusters() {
        // Create data with 2 clear clusters
        let mat = DMatrix::from_row_slice(
            2,
            6,
            &[
                0.0, 0.1, 0.2, 10.0, 10.1, 10.2, // row 0
                0.0, 0.1, 0.0, 10.0, 10.1, 10.2, // row 1
            ],
        );

        let args = KmeansArgs::with_clusters(2);
        let membership = mat.kmeans_columns(args);

        assert_eq!(membership.len(), 6);

        // First 3 columns should be in same cluster
        assert_eq!(membership[0], membership[1]);
        assert_eq!(membership[1], membership[2]);

        // Last 3 columns should be in same cluster
        assert_eq!(membership[3], membership[4]);
        assert_eq!(membership[4], membership[5]);

        // Two groups should be different
        assert_ne!(membership[0], membership[3]);
    }

    #[test]
    fn test_kmeans_rows() {
        // Create data with 2 clear row clusters
        let mat = DMatrix::from_row_slice(
            4,
            2,
            &[
                0.0, 0.0, // row 0 - cluster A
                0.1, 0.1, // row 1 - cluster A
                10.0, 10.0, // row 2 - cluster B
                10.1, 10.1, // row 3 - cluster B
            ],
        );

        let args = KmeansArgs::with_clusters(2);
        let membership = mat.kmeans_rows(args);

        assert_eq!(membership.len(), 4);

        // First 2 rows should be in same cluster
        assert_eq!(membership[0], membership[1]);

        // Last 2 rows should be in same cluster
        assert_eq!(membership[2], membership[3]);

        // Two groups should be different
        assert_ne!(membership[0], membership[2]);
    }

    #[test]
    fn test_kmeans_empty_matrix() {
        let mat: DMatrix<f32> = DMatrix::zeros(0, 0);

        let col_membership = mat.kmeans_columns(KmeansArgs::with_clusters(2));
        let row_membership = mat.kmeans_rows(KmeansArgs::with_clusters(2));

        assert!(col_membership.is_empty());
        assert!(row_membership.is_empty());
    }
}
