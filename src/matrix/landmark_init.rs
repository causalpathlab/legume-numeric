//! A 2D start for UMAP-style SGD that unrolls what PCA folds.
//!
//! The PCA start projects a curved manifold onto its two widest
//! directions, so the turns of a roll, or the branches of a tree, start on
//! top of one another; the SGD then pulls each point toward its graph
//! neighbours from a tangle it rarely undoes. This start lays out
//! distances along the kNN graph instead:
//!
//! 1. k-means++ landmarks in feature space, each point labelled by its
//!    nearest;
//! 2. two landmarks are linked when at least [`MIN_LINKS`] graph edges join
//!    their points, at the distance between them, and the geodesics are the
//!    shortest paths over these links;
//! 3. classical MDS of the geodesics, each landmark weighted by the points
//!    it stands for;
//! 4. each point starts at its landmark, jittered across half the spacing
//!    between landmarks.
//!
//! Landmarks no path joins (graph components) are kept apart by their
//! straight distance, which a geodesic can never undercut. With no edge
//! between landmarks at all, the weighted MDS of straight distances is the
//! PCA of the landmarks counted by their points, close to the PCA of the
//! points, so separate clusters start as PCA would place them.
//!
//! References:
//! - Tenenbaum, de Silva & Langford, *Science* 2000: Isomap.
//! - de Silva & Tenenbaum, *NeurIPS* 2002: landmark Isomap.
//! - Kobak & Linderman, *Nat. Biotechnol.* 2021: the start shapes the
//!   global structure UMAP and t-SNE keep.

use crate::matrix::kmeans::kmeans_centroids_seeded;
use crate::matrix::pca::{random_init_2d, INIT_SCALE};
use nalgebra::DMatrix;
use rand::{rngs::SmallRng, RngExt, SeedableRng};
use rayon::prelude::*;

type Mat = DMatrix<f32>;

/// Landmarks a start lays out by default. Each landmark's points must lie
/// closer together than the gap between two folds, or its cell straddles
/// the fold and links across it: a Swiss roll that 300 landmarks unroll
/// stays folded with 30, however many points there are. The geodesics and
/// the MDS cost `O(k³)` in the landmark count.
pub const N_LANDMARKS: usize = 300;

/// Graph edges it takes to link two landmarks. A noisy kNN graph has the
/// odd edge between far parts of a manifold; a single one would make a
/// geodesic shortcut across a fold, a few between the same two landmarks
/// rarely come by chance.
pub const MIN_LINKS: usize = 3;

/// Lloyd iterations for the landmarks.
const KMEANS_ITER: usize = 30;

/// Landmarks closer than this fraction of [`INIT_SCALE`] in the plane are on
/// top of one another: rounding leaves them a hair apart, and a jitter that
/// small would be lost on coordinates near [`INIT_SCALE`].
const COINCIDENT: f32 = 1e-3;

/// Row-major `n × 2` start for the layout of the rows of `features` on
/// their kNN graph `edges` (`(i, j, weight)`, each pair once, weights
/// unused), with up to `n_landmarks` landmarks, scaled so the largest
/// `|coordinate|` is about [`INIT_SCALE`]. Pass the features the graph was
/// built on, or their leading PC scores. Falls back to [`random_init_2d`]
/// when the rows give fewer than two distinct landmarks.
///
/// # Panics
/// If an edge names a point at or past `features.nrows()`.
#[must_use]
pub fn landmark_init_2d(
    features: &Mat,
    edges: &[(usize, usize, f32)],
    n_landmarks: usize,
    seed: u64,
) -> Vec<f32> {
    let n = features.nrows();
    let (centroids, labels) =
        kmeans_centroids_seeded(features, n_landmarks.min(n), KMEANS_ITER, seed);
    let (centroids, labels, weights) = drop_empty(&centroids, &labels);
    let k = centroids.nrows();
    if k < 2 {
        return random_init_2d(n, seed);
    }
    let geodesics = landmark_geodesics(&centroids, &labels, edges);
    let mut pos = weighted_mds_2d(&geodesics, &weights);
    let reach = pos.iter().flatten().fold(0.0_f32, |m, &v| m.max(v.abs()));
    if reach <= 0.0 || !reach.is_finite() {
        return random_init_2d(n, seed);
    }
    pos.iter_mut()
        .flatten()
        .for_each(|v| *v = *v * INIT_SCALE / reach);

    // The mean distance from a landmark to the nearest one not on top of it:
    // landmarks apart in feature space can meet in the plane, and points
    // must not start at zero distance, where the SGD cannot part them. The
    // MDS is centred and reaches `INIT_SCALE`, so no landmark has all the
    // others on top of it.
    let spacing = (0..k)
        .map(|a| {
            (0..k)
                .map(|b| (pos[a][0] - pos[b][0]).hypot(pos[a][1] - pos[b][1]))
                .filter(|&d| d > COINCIDENT * INIT_SCALE)
                .fold(f32::INFINITY, f32::min)
        })
        .sum::<f32>()
        / k as f32;
    let half = 0.5 * spacing;
    let mut rng = SmallRng::seed_from_u64(seed ^ 0x1a4d_3a2c_5eed_0001);
    labels
        .iter()
        .flat_map(|&l| pos[l])
        .map(|v| v + rng.random_range(-0.5..0.5) * half)
        .collect()
}

/// The landmarks some point is nearest to, the labels renumbered to them,
/// and how many points each stands for.
fn drop_empty(centroids: &Mat, labels: &[usize]) -> (Mat, Vec<usize>, Vec<f32>) {
    let mut count = vec![0_usize; centroids.nrows()];
    for &l in labels {
        count[l] += 1;
    }
    let kept: Vec<usize> = (0..count.len()).filter(|&l| count[l] > 0).collect();
    let mut renumber = vec![usize::MAX; count.len()];
    for (new, &old) in kept.iter().enumerate() {
        renumber[old] = new;
    }
    let centroids = centroids.select_rows(&kept);
    let labels = labels.iter().map(|&l| renumber[l]).collect();
    let weights = kept.iter().map(|&l| count[l] as f32).collect();
    (centroids, labels, weights)
}

/// `k × k` distances between landmarks along their links (Floyd-Warshall),
/// or straight across where no path joins them.
fn landmark_geodesics(centroids: &Mat, labels: &[usize], edges: &[(usize, usize, f32)]) -> Mat {
    let k = centroids.nrows();
    let straight = |a: usize, b: usize| (centroids.row(a) - centroids.row(b)).norm();
    let mut links = vec![0_u32; k * k];
    for &(i, j, _) in edges {
        let (a, b) = (labels[i], labels[j]);
        if a != b {
            links[a * k + b] += 1;
            links[b * k + a] += 1;
        }
    }
    // Row-major `g[a * k + b]`.
    let mut g: Vec<f32> = (0..k * k)
        .map(|ab| {
            let (a, b) = (ab / k, ab % k);
            if a == b {
                0.0
            } else if links[ab] as usize >= MIN_LINKS {
                straight(a, b)
            } else {
                f32::INFINITY
            }
        })
        .collect();
    let mut via = vec![0.0_f32; k];
    for m in 0..k {
        via.copy_from_slice(&g[m * k..(m + 1) * k]);
        g.par_chunks_mut(k).for_each(|row| {
            let to_m = row[m];
            if to_m.is_finite() {
                for (d, &v) in row.iter_mut().zip(&via) {
                    *d = d.min(to_m + v);
                }
            }
        });
    }
    for (ab, d) in g.iter_mut().enumerate() {
        if !d.is_finite() {
            *d = straight(ab / k, ab % k);
        }
    }
    // `g` is symmetric, so its row-major layout reads the same column-major.
    Mat::from_vec(k, k, g)
}

/// Classical MDS of the distances `d`, landmark `a` counted `w[a]` times:
/// the 2D coordinates whose `w`-weighted inner products best match the
/// doubly centred `-d²/2`. With straight distances this is the PCA of the
/// landmarks counted by their points. Needs at least two landmarks.
fn weighted_mds_2d(d: &Mat, w: &[f32]) -> Vec<[f32; 2]> {
    let k = d.nrows();
    let total: f32 = w.iter().sum();
    let w: Vec<f32> = w.iter().map(|&x| x / total).collect();
    let d2 = d.map(|x| x * x);
    let mean: Vec<f32> = (0..k)
        .map(|a| (0..k).map(|b| w[b] * d2[(a, b)]).sum())
        .collect();
    let grand: f32 = (0..k).map(|a| w[a] * mean[a]).sum();
    let gram = Mat::from_fn(k, k, |a, b| {
        -0.5 * (d2[(a, b)] - mean[a] - mean[b] + grand) * (w[a] * w[b]).sqrt()
    });
    let eig = gram.symmetric_eigen();
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&x, &y| eig.eigenvalues[y].total_cmp(&eig.eigenvalues[x]));
    let axis = |c: usize| {
        let e = order[c];
        (e, eig.eigenvalues[e].max(0.0).sqrt())
    };
    let ((e0, s0), (e1, s1)) = (axis(0), axis(1));
    (0..k)
        .map(|a| {
            let unweigh = w[a].sqrt();
            [
                eig.eigenvectors[(a, e0)] * s0 / unweigh,
                eig.eigenvectors[(a, e1)] * s1 / unweigh,
            ]
        })
        .collect()
}

#[cfg(test)]
pub(crate) mod tests;
