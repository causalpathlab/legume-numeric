//! Minimal UMAP-style SGD layout over a weighted edge list. Expects a
//! pre-built fuzzy kNN graph (edges + `[0, 1]` weights); the low-d kernel
//! is `1 / (1 + a·d^(2b))` (see [`Umap::a`]); [`Umap::with_min_dist`]
//! fits it to a `min_dist` and `spread`.
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
//! [`Umap::fit_on`] runs the same SGD on a candle device (Metal, CUDA or
//! CPU) with synchronous epochs; start it from PCA.
//!
//! The start sets the global shape the SGD keeps. On a curved manifold
//! the PCA start folds it, and
//! [`crate::matrix::landmark_init::landmark_init_2d`] lays it out along the
//! graph instead.
//!
//! References:
//! - McInnes, Healy & Melville, *arXiv* 1802.03426 — UMAP.
//! - Recht et al., *NeurIPS* 2011 — HOGWILD! lock-free SGD.

use nalgebra::Vector2;
use rand::{rngs::SmallRng, RngExt, SeedableRng};
use rayon::prelude::*;

#[cfg(feature = "tensor")]
mod device;

const A: f32 = 1.929;
const B: f32 = 0.7915;

#[derive(Debug, Clone)]
pub struct Umap {
    pub n_epochs: usize,
    pub negative_sample_rate: usize,
    pub learning_rate: f32,
    pub seed: u64,
    /// Low-d kernel `1/(1 + a·d^(2b))`. Default `(1.929, 0.7915)`, UMAP's
    /// fit for spread=1, min_dist=0.001 (see [`Umap::with_min_dist`]).
    /// `(1.0, 1.0)` = **t-UMAP** (`uwot::tumap`): the pure t-distribution
    /// kernel `1/(1+d²)`, whose heavier tails give gentler attraction and a
    /// more spread-out layout — better for continuum/branch structure. Use
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

/// umap-learn's default `min_dist`, for [`Umap::with_min_dist`]; not
/// [`Umap::default`]'s kernel (see [`Umap::a`]).
pub const MIN_DIST: f32 = 0.1;
/// umap-learn's default `spread`, for [`Umap::with_min_dist`].
pub const SPREAD: f32 = 1.0;

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

    /// This SGD with UMAP's kernel for `min_dist` (how tightly points may
    /// pack) and `spread` (the scale of the map's distances), `(a, b)`
    /// fitted to them as umap-learn's `find_ab_params` fits them. Errs
    /// unless `spread` is positive and finite and `min_dist` in
    /// `[0, spread]`.
    pub fn with_min_dist(self, min_dist: f32, spread: f32) -> anyhow::Result<Self> {
        anyhow::ensure!(
            spread.is_finite() && spread > 0.0,
            "UMAP's spread must be positive and finite, not {spread}"
        );
        anyhow::ensure!(
            (0.0..=spread).contains(&min_dist),
            "UMAP's min_dist must lie in [0, spread = {spread}], not {min_dist}"
        );
        let (a, b) = fit_ab(min_dist, spread);
        Ok(Self { a, b, ..self })
    }

    /// uwot's epochs for `n` points: 500 up to 10,000, else 200.
    #[must_use]
    pub fn default_epochs(n: usize) -> usize {
        if n <= 10_000 {
            500
        } else {
            200
        }
    }
}

/// UMAP's kernel `1/(1 + a x^(2b))` least-squares fitted to the target
/// curve, `1` below `min_dist` and `exp(-(x - min_dist)/spread)` above, at
/// 300 points of `[0, 3 spread]`: umap-learn's `find_ab_params`, here by
/// Levenberg-Marquardt from `(1, 1)` as scipy's `curve_fit` does.
fn fit_ab(min_dist: f32, spread: f32) -> (f32, f32) {
    let (min_dist, spread) = (f64::from(min_dist), f64::from(spread));
    let xs: Vec<f64> = (0..300)
        .map(|i| 3.0 * spread * f64::from(i) / 299.0)
        .collect();
    let target = |x: f64| {
        if x < min_dist {
            1.0
        } else {
            (-(x - min_dist) / spread).exp()
        }
    };
    let sse = |a: f64, b: f64| -> f64 {
        xs.iter()
            .map(|&x| (1.0 / (1.0 + a * x.powf(2.0 * b)) - target(x)).powi(2))
            .sum()
    };
    let (mut a, mut b, mut lambda) = (1.0_f64, 1.0_f64, 1e-3_f64);
    let mut err = sse(a, b);
    for _ in 0..500 {
        // The normal equations of the residuals' Jacobian in (a, b).
        let (mut jaa, mut jab, mut jbb, mut ga, mut gb) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for &x in &xs {
            let p = x.powf(2.0 * b);
            let m = 1.0 / (1.0 + a * p);
            let r = m - target(x);
            let da = -p * m * m;
            // x^(2b) ln x vanishes at x = 0.
            let db = if x > 0.0 {
                -a * p * 2.0 * x.ln() * m * m
            } else {
                0.0
            };
            jaa += da * da;
            jab += da * db;
            jbb += db * db;
            ga += da * r;
            gb += db * r;
        }
        // Damp until a step lowers the error; none does at the minimum.
        let mut stepped = false;
        while lambda < 1e12 {
            let (haa, hbb) = (jaa * (1.0 + lambda), jbb * (1.0 + lambda));
            let det = haa * hbb - jab * jab;
            let (na, nb) = (
                a - (hbb * ga - jab * gb) / det,
                b - (haa * gb - jab * ga) / det,
            );
            if na > 0.0 && nb > 0.0 && det.is_finite() {
                let next = sse(na, nb);
                if next < err {
                    stepped = err - next > 1e-15 * err;
                    (a, b, err, lambda) = (na, nb, next, lambda / 10.0);
                    break;
                }
            }
            lambda *= 10.0;
        }
        if !stepped {
            break;
        }
    }
    (a as f32, b as f32)
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
    let grad = (diff * attraction_coeff(d2, a, b)).map(clamp4) * alpha;
    y.add(i, grad);
    y.add(j, -grad);
}

#[inline]
fn apply_repulsion(y: &HogwildCoords, i: usize, k: usize, alpha: f32, a: f32, b: f32) {
    let diff = y.get(i) - y.get(k);
    let d2 = diff.norm_squared();
    let coeff = if d2 > 0.0 {
        repulsion_coeff(d2, a, b)
    } else {
        4.0
    };
    let grad = (diff * coeff).map(clamp4) * alpha;
    y.add(i, grad);
}

/// `x^b`, sparing the `powf` for t-UMAP's `b = 1`.
#[inline]
fn pow_b(x: f32, b: f32) -> f32 {
    if b == 1.0 {
        x
    } else {
        x.powf(b)
    }
}

/// The attraction gradient's coefficient at squared distance `d2 > 0`.
#[inline]
fn attraction_coeff(d2: f32, a: f32, b: f32) -> f32 {
    let d2b = pow_b(d2, b);
    -2.0 * a * b * (d2b / d2) / (a * d2b + 1.0)
}

/// The repulsion gradient's coefficient at squared distance `d2 > 0`.
#[inline]
fn repulsion_coeff(d2: f32, a: f32, b: f32) -> f32 {
    2.0 * b / ((0.001 + d2) * (a * pow_b(d2, b) + 1.0))
}

#[inline]
fn clamp4(x: f32) -> f32 {
    x.clamp(-4.0, 4.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::test_support::fuzzy_edges;
    use nalgebra::DMatrix;

    /// The fuzzy kNN graph of uniform points in a square, and a random
    /// start for its layout.
    pub(super) fn square_graph(n: usize) -> (Vec<(usize, usize, f32)>, Vec<f32>) {
        let mut rng = SmallRng::seed_from_u64(3);
        let data = DMatrix::from_fn(n, 2, |_, _| rng.random_range(0.0_f32..1.0));
        let edges = fuzzy_edges(&data, 14);
        let init: Vec<f32> = (0..n * 2).map(|_| rng.random_range(-10.0..10.0)).collect();
        (edges, init)
    }

    /// Uniform points in a square, laid out by t-UMAP on their kNN graph.
    fn layout(n: usize) -> Vec<f32> {
        let (edges, init) = square_graph(n);
        Umap {
            n_epochs: 200,
            ..Umap::tumap()
        }
        .fit(&edges, n, &init)
    }

    /// Mean distance to the nearest other point, for the points `of`.
    pub(super) fn spacing(y: &[f32], n: usize, of: std::ops::Range<usize>) -> f32 {
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

    #[test]
    fn min_dist_and_spread_fit_the_kernel_as_umap_learn_does() {
        // umap-learn's find_ab_params (scipy's curve_fit) at these
        // (min_dist, spread).
        for (min_dist, spread, a, b) in [
            (0.1, 1.0, 1.576_943_5, 0.895_060_9),
            (0.5, 1.0, 0.583_03, 1.334_167),
            (0.01, 1.0, 1.895_605_9, 0.800_637_8),
            (0.5, 2.0, 0.258_878_7, 1.057_5),
            (0.99, 1.0, 0.119_305_3, 1.916_390_4),
        ] {
            let u = Umap::tumap().with_min_dist(min_dist, spread).unwrap();
            let near = |x: f32, y: f32| (x - y).abs() < 1e-3 * y;
            assert!(
                near(u.a, a) && near(u.b, b),
                "min_dist {min_dist}, spread {spread}: ({}, {}), not ({a}, {b})",
                u.a,
                u.b
            );
            assert_eq!(u.n_epochs, Umap::tumap().n_epochs);
        }
    }

    #[test]
    fn a_min_dist_or_spread_out_of_range_is_refused() {
        for (min_dist, spread) in [
            (-0.1, 1.0),
            (0.1, 0.0),
            (1.5, 1.0),
            (0.1, f32::INFINITY),
            (f32::NAN, 1.0),
        ] {
            assert!(
                Umap::default().with_min_dist(min_dist, spread).is_err(),
                "({min_dist}, {spread})"
            );
        }
    }

    #[test]
    fn epochs_follow_uwot() {
        assert_eq!(Umap::default_epochs(10_000), 500);
        assert_eq!(Umap::default_epochs(10_001), 200);
    }
}
