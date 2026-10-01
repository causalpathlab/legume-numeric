//! Minimal UMAP-style SGD layout over a weighted edge list. Expects a
//! pre-built fuzzy kNN graph (edges + `[0, 1]` weights); the low-d kernel
//! is `1 / (1 + a·d^(2b))` with the standard `(a, b) ≈ (1.929, 0.7915)`
//! fit for `spread=1, min_dist=0.1`.
//!
//! Inner math uses `Vector2<f32>` (SIMD auto-vec). Edges are processed
//! in parallel via rayon with HOGWILD! benign races on the shared
//! coords buffer — UMAP's SGD is robust to these; the reference numba
//! impl does the same.
//!
//! Pair this with [`crate::matrix::knn_graph::KnnGraph::from_columns_fuzzy`]
//! to build the input edge list. Shared by `senna layout umap` and
//! `senna lineage-plot`.
//!
//! As uwot and umap-learn, each undirected edge is sampled from both ends:
//! each end in turn pulls the pair together and is pushed away from
//! `negative_sample_rate` random points. The result is centred.
//!
//! References:
//! - McInnes, Healy & Melville, *arXiv* 1802.03426 — UMAP.
//! - Recht et al., *NeurIPS* 2011 — HOGWILD! lock-free SGD.

use nalgebra::Vector2;
use rand::{rngs::SmallRng, RngExt, SeedableRng};
use rayon::prelude::*;

const A: f32 = 1.929;
const B: f32 = 0.7915;

pub struct Umap {
    pub n_epochs: usize,
    pub negative_sample_rate: usize,
    pub learning_rate: f32,
    pub seed: u64,
    /// Low-d kernel `1/(1 + a·d^(2b))`. Default `(1.929, 0.7915)` = standard UMAP
    /// (spread=1, min_dist=0.1). `(1.0, 1.0)` = **t-UMAP** (`uwot::tumap`): the pure
    /// t-distribution kernel `1/(1+d²)`, whose heavier tails give gentler attraction
    /// and a more spread-out layout — better for continuum/branch structure. Use
    /// [`Umap::tumap`] for that.
    pub a: f32,
    pub b: f32,
}

impl Default for Umap {
    fn default() -> Self {
        Self {
            n_epochs: 500,
            negative_sample_rate: 5,
            learning_rate: 1.0,
            seed: 42,
            a: A,
            b: B,
        }
    }
}

impl Umap {
    /// t-UMAP (`uwot::tumap`): the `a=b=1` kernel `1/(1+d²)` — more spread than the
    /// default standard-UMAP kernel. Other fields keep their defaults.
    #[must_use]
    pub fn tumap() -> Self {
        Self {
            a: 1.0,
            b: 1.0,
            ..Self::default()
        }
    }
}

/// Shared handle for HOGWILD! parallel SGD on `coords`.
struct HogwildCoords {
    ptr: *mut f32,
    n: usize,
}

unsafe impl Sync for HogwildCoords {}
unsafe impl Send for HogwildCoords {}

impl HogwildCoords {
    #[inline]
    fn get(&self, i: usize) -> Vector2<f32> {
        debug_assert!(i < self.n);
        // SAFETY: HOGWILD! allows benign races; each index is 2 f32s.
        unsafe { Vector2::new(*self.ptr.add(i * 2), *self.ptr.add(i * 2 + 1)) }
    }

    #[inline]
    fn add(&self, i: usize, delta: Vector2<f32>) {
        debug_assert!(i < self.n);
        // SAFETY: HOGWILD! tolerates torn updates; values are bounded by clamp.
        unsafe {
            *self.ptr.add(i * 2) += delta.x;
            *self.ptr.add(i * 2 + 1) += delta.y;
        }
    }
}

impl Umap {
    /// Run HOGWILD! SGD on the given undirected edge list.
    ///
    /// * `edges`: `(i, j, weight)` with `weight ∈ (0, 1]`, each pair once.
    /// * `n` — number of points (rows in `init`/output).
    /// * `init` — row-major `n × 2` initial coords.
    pub fn fit(&self, edges: &[(usize, usize, f32)], n: usize, init: &[f32]) -> Vec<f32> {
        assert_eq!(init.len(), n * 2, "init size mismatch");
        let mut y = init.to_vec();

        let eps = 1e-4_f32;
        let max_weight = edges.iter().map(|e| e.2).fold(0.0_f32, f32::max).max(eps);
        let epochs_per_sample: Vec<f32> = edges
            .iter()
            .map(|&(_, _, w)| {
                if w > 0.0 {
                    max_weight / w
                } else {
                    f32::INFINITY
                }
            })
            .collect();
        let mut next_epoch: Vec<f32> = epochs_per_sample.clone();

        let coords = HogwildCoords {
            ptr: y.as_mut_ptr(),
            n,
        };
        let coords = &coords;
        let n_neg = self.negative_sample_rate;
        let seed = self.seed;
        let (a, b) = (self.a, self.b);

        for epoch in 0..self.n_epochs {
            let epoch_f = epoch as f32;
            let alpha = self.learning_rate * (1.0 - epoch_f / self.n_epochs as f32);

            edges
                .par_iter()
                .zip(next_epoch.par_iter_mut())
                .enumerate()
                .for_each_init(
                    || {
                        let tid = rayon::current_thread_index().unwrap_or(0) as u64;
                        SmallRng::seed_from_u64(seed ^ ((epoch as u64) << 32) ^ tid)
                    },
                    |rng, (e_idx, (&(i, j, _), ne))| {
                        if *ne > epoch_f {
                            return;
                        }
                        // Both directions of the symmetric graph, as uwot
                        // samples them: each end is a head once.
                        for (head, tail) in [(i, j), (j, i)] {
                            apply_attraction(coords, head, tail, alpha, a, b);
                            for _ in 0..n_neg {
                                let k = rng.random_range(0..n);
                                if k == head {
                                    continue;
                                }
                                apply_repulsion(coords, head, k, alpha, a, b);
                            }
                        }

                        *ne += epochs_per_sample[e_idx];
                    },
                );
        }

        // Centred, as uwot returns it.
        for c in 0..2 {
            let mean = (0..n).map(|i| y[i * 2 + c]).sum::<f32>() / n.max(1) as f32;
            (0..n).for_each(|i| y[i * 2 + c] -= mean);
        }
        y
    }
}

#[inline]
fn apply_attraction(y: &HogwildCoords, i: usize, j: usize, alpha: f32, a: f32, b: f32) {
    let diff = y.get(i) - y.get(j);
    let d2 = diff.norm_squared();
    if d2 <= 0.0 {
        return;
    }
    let d2b = d2.powf(b);
    let coeff = -2.0 * a * b * (d2b / d2) / (a * d2b + 1.0);
    let grad = (diff * coeff).map(clamp4) * alpha;
    y.add(i, grad);
    y.add(j, -grad);
}

#[inline]
fn apply_repulsion(y: &HogwildCoords, i: usize, k: usize, alpha: f32, a: f32, b: f32) {
    let diff = y.get(i) - y.get(k);
    let d2 = diff.norm_squared();
    let coeff = if d2 > 0.0 {
        2.0 * b / ((0.001 + d2) * (a * d2.powf(b) + 1.0))
    } else {
        4.0
    };
    let grad = (diff * coeff).map(clamp4) * alpha;
    y.add(i, grad);
}

#[inline]
fn clamp4(x: f32) -> f32 {
    x.clamp(-4.0, 4.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::knn_graph::{KnnGraph, KnnGraphArgs};
    use nalgebra::DMatrix;

    /// Uniform points in a square, laid out by t-UMAP on their kNN graph.
    fn layout(n: usize) -> Vec<f32> {
        let mut rng = SmallRng::seed_from_u64(3);
        let data = DMatrix::from_fn(n, 2, |_, _| rng.random_range(0.0_f32..1.0));
        let (graph, w) = KnnGraph::from_rows_fuzzy(
            &data,
            KnnGraphArgs {
                knn: 14,
                block_size: 256,
                reciprocal: false,
            },
        )
        .unwrap();
        let edges: Vec<_> = graph
            .edges
            .iter()
            .zip(&w)
            .map(|(&(i, j), &w)| (i, j, w))
            .collect();
        let init: Vec<f32> = (0..n * 2).map(|_| rng.random_range(-10.0..10.0)).collect();
        Umap {
            n_epochs: 200,
            ..Umap::tumap()
        }
        .fit(&edges, n, &init)
    }

    /// Mean distance to the nearest other point, for the points `of`.
    fn spacing(y: &[f32], n: usize, of: std::ops::Range<usize>) -> f32 {
        let at = |i: usize| Vector2::new(y[i * 2], y[i * 2 + 1]);
        let m = of.len() as f32;
        of.map(|i| {
            (0..n)
                .filter(|&j| j != i)
                .map(|j| (at(i) - at(j)).norm())
                .fold(f32::INFINITY, f32::min)
        })
        .sum::<f32>()
            / m
    }

    #[test]
    fn points_are_spaced_alike_whatever_their_index() {
        // Every edge is stored once as (low, high): sampled from that end
        // only, high-index points were pushed apart less and bunched (a
        // ratio near 0.84 here).
        let n = 1500;
        let y = layout(n);
        let (low, high) = (spacing(&y, n, 0..n / 2), spacing(&y, n, n / 2..n));
        let ratio = high / low;
        assert!((0.9..1.1).contains(&ratio), "high/low spacing {ratio}");
    }

    #[test]
    fn the_layout_is_centred() {
        let n = 300;
        let y = layout(n);
        for c in 0..2 {
            let mean = (0..n).map(|i| y[i * 2 + c]).sum::<f32>() / n as f32;
            assert!(mean.abs() < 1e-3, "mean {mean}");
        }
    }
}
