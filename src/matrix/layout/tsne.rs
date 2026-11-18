//! t-SNE of a weighted graph: [`TSne`] and [`tsne_layout_2d_on_graph`].
//!
//! The gradient is the exact one, written out (van der Maaten & Hinton
//! 2008, eq. 5, plus the densMAP term's), over every pair: O(n²) per
//! iteration in parallel, meant for the PB graphs a layout coarsens to.

use super::Mat;
use candle_core::Result;
use rayon::prelude::*;

/// Smallest adaptive gain.
const MIN_GAIN: f32 = 0.01;

/// Args of [`tsne_layout_2d_on_graph`].
#[derive(Debug, Clone)]
pub struct TsneArgs {
    /// Gradient-descent iterations.
    pub n_iter: usize,
    /// densMAP-style density preservation strength (0 disables it); needs
    /// `weights`.
    pub density_lambda: f32,
    /// Per-node weights (PB cell counts), the density target.
    pub weights: Option<Vec<usize>>,
    /// Seed of the random start ([`crate::matrix::pca::random_init_2d`]).
    pub seed: u64,
}

impl Default for TsneArgs {
    fn default() -> Self {
        Self {
            n_iter: 1000,
            density_lambda: 0.0,
            weights: None,
            seed: 42,
        }
    }
}

/// t-SNE of the `n` nodes of the undirected graph `edges` (`(i, j, weight)`,
/// weights affinities), from a seeded random start, as [`TSne::fit_graph`].
/// Returns `n × 2`.
pub fn tsne_layout_2d_on_graph(
    n: usize,
    edges: &[(usize, usize, f32)],
    args: &TsneArgs,
) -> anyhow::Result<Mat> {
    let mut tsne = TSne::default()
        .n_iter(args.n_iter)
        .density_lambda(args.density_lambda);
    if let Some(w) = &args.weights {
        tsne = tsne.weights(w);
    }
    let init = crate::matrix::pca::random_init_2d(n, args.seed);
    let y = tsne.fit_graph(n, edges, &init)?;
    Ok(Mat::from_row_slice(n, 2, &y))
}

/// t-SNE with an Rtsne-aligned update rule: per-parameter adaptive gains
/// (Jacobs 1988), momentum switching, strong early exaggeration, and init
/// rescaling to std = 1e-4, on a graph's affinities. Also supports a
/// densMAP-style density-preservation auxiliary loss.
///
/// References:
/// - van der Maaten & Hinton, *JMLR* 2008 — t-SNE.
/// - Jacobs, *Neural Networks* 1988 — increased rate of convergence through
///   learning rate adaptation (delta-bar-delta / adaptive gains).
/// - Narayan, Berger & Cho, *Nat Biotechnol* 2021 — densMAP / den-SNE
///   density-preserving extension.
#[derive(Debug, Clone)]
pub struct TSne {
    learning_rate: f32,
    initial_momentum: f32,
    final_momentum: f32,
    n_iter: usize,
    early_exaggeration: f32,
    early_exaggeration_iter: usize,
    /// Optional per-point weights (PB cell counts) for the densMAP target.
    weights: Option<Vec<f32>>,
    /// densMAP-style density-preservation strength. When `> 0` and
    /// `weights` is set, an auxiliary loss pins each point's soft low-D
    /// local radius `R_i = Σ_j P_ij · d_ij(Y)` to `∝ 1/√n_i`, so dense
    /// regions (big PBs) shrink and sparse regions (small PBs) spread.
    density_lambda: f32,
}

impl Default for TSne {
    fn default() -> Self {
        Self {
            learning_rate: 200.0,
            initial_momentum: 0.5,
            final_momentum: 0.8,
            n_iter: 1000,
            early_exaggeration: 12.0,
            early_exaggeration_iter: 250,
            weights: None,
            density_lambda: 0.0,
        }
    }
}

impl TSne {
    #[must_use]
    pub fn n_iter(mut self, n: usize) -> Self {
        self.n_iter = n;
        self
    }

    /// Attach per-point weights (PB cell counts), the density target of
    /// [`TSne::density_lambda`]. A graph's summed edge weights already grow
    /// with its nodes' sizes, so P itself is not weighted again.
    #[must_use]
    pub fn weights(mut self, w: &[usize]) -> Self {
        self.weights = Some(w.iter().map(|&x| x as f32).collect());
        self
    }

    /// Enable densMAP-style density preservation with strength `lambda`.
    /// Requires `weights` to be set — the density target is `1/√w_i`.
    /// A good starting value is ~0.1. Set to 0 to disable.
    #[must_use]
    pub fn density_lambda(mut self, lambda: f32) -> Self {
        self.density_lambda = lambda.max(0.0);
        self
    }

    /// t-SNE of the `n` nodes of the undirected graph `edges`
    /// (`(i, j, weight)`), from `init` (`n × 2`, row-major). Each node's edge
    /// weights, normalized, are its `P(j|i)`: the graph is taken as it is
    /// rather than through a perplexity-calibrated Gaussian of distances,
    /// which on a kNN graph has too little contrast between linked and
    /// unlinked nodes to keep groups apart. Returns a flat n×2 embedding.
    ///
    /// The update rule follows Rtsne / van der Maaten's reference
    /// implementation: adaptive per-parameter gains, momentum switch at
    /// `early_exaggeration_iter`, init rescaled to std = 1e-4 so early
    /// exaggeration has room to separate clusters, and Y centered every
    /// iteration to prevent drift.
    pub fn fit_graph(
        &self,
        n: usize,
        edges: &[(usize, usize, f32)],
        init: &[f32],
    ) -> Result<Vec<f32>> {
        let p = joint_probabilities(n, edges);
        let p_sum: f32 = p.iter().sum();

        // The centered log-target of density preservation:
        //   target_i ∝ 1/√w_i   =>   log target_i = -½ log w_i
        let density_target: Option<Vec<f32>> = if self.density_lambda > 0.0 {
            self.weights.as_ref().filter(|w| w.len() == n).map(|w| {
                let mut log_t: Vec<f32> = w.iter().map(|&wi| -0.5 * wi.max(1.0).ln()).collect();
                let mean = log_t.iter().sum::<f32>() / n as f32;
                for v in &mut log_t {
                    *v -= mean;
                }
                log_t
            })
        } else {
            None
        };

        let mut y: Vec<f32> = init.to_vec();

        // Rescale the init to std = 1e-4 (Rtsne / openTSNE convention).
        // Keeps the *shape* of the init while shrinking the scale so that
        // early exaggeration has room to push clusters apart.
        center(&mut y);
        let var: f32 = y.iter().map(|v| v * v).sum::<f32>() / (y.len() as f32).max(1.0);
        let scale = 1e-4 / var.sqrt().max(1e-12);
        for v in &mut y {
            *v *= scale;
        }

        // Auto-scale learning rate with n (Kobak & Berens 2019). The fixed
        // lr=200 Rtsne convention diverges for small n: early-exaggeration
        // gradients grow faster than the Student-t tail can damp them,
        // blowing Y up to overflow within ~200 iters at n ≈ 70.
        let lr_auto = (n as f32 / self.early_exaggeration).max(50.0);
        let learning_rate = self.learning_rate.min(lr_auto);

        // Per-parameter velocity and adaptive gains (Jacobs 1988).
        let mut velocity: Vec<f32> = vec![0.0; n * 2];
        let mut gains: Vec<f32> = vec![1.0; n * 2];

        for iter in 0..self.n_iter {
            let early = iter < self.early_exaggeration_iter;
            let p_mult = if early { self.early_exaggeration } else { 1.0 };
            // Density term disabled during early exaggeration so gradients
            // through log R don't blow up at tiny distances.
            let density = density_target
                .as_deref()
                .filter(|_| !early)
                .map(|t| (t, self.density_lambda));
            let g = gradient(&y, &p, p_mult, p_sum, density);

            // Adaptive gains (van der Maaten reference):
            //   sign(grad) ≠ sign(velocity) (oscillating) → gain += 0.2
            //   sign(grad) == sign(velocity) (steady)      → gain *= 0.8
            for (gain, (&v, &gi)) in gains.iter_mut().zip(velocity.iter().zip(&g)) {
                if (v > 0.0) == (gi > 0.0) {
                    *gain *= 0.8;
                } else {
                    *gain += 0.2;
                }
                *gain = gain.max(MIN_GAIN);
            }

            // Momentum switch at the end of early exaggeration.
            let momentum = if early {
                self.initial_momentum
            } else {
                self.final_momentum
            };

            // v ← momentum · v − lr · gains · grad; Y ← Y + v, centered so
            // the mean stays at the origin.
            for i in 0..velocity.len() {
                velocity[i] = momentum * velocity[i] - learning_rate * gains[i] * g[i];
                y[i] += velocity[i];
            }
            center(&mut y);
        }

        Ok(y)
    }
}

/// Move the points of `y` (flat `n × 2`) so their mean is the origin.
fn center(y: &mut [f32]) {
    let n = y.len() / 2;
    let (sx, sy) = (0..n).fold((0.0, 0.0), |(a, b), i| (a + y[2 * i], b + y[2 * i + 1]));
    let (mx, my) = (sx / n.max(1) as f32, sy / n.max(1) as f32);
    for i in 0..n {
        y[2 * i] -= mx;
        y[2 * i + 1] -= my;
    }
}

/// The exact gradient (flat `n × 2`) at `y` of `KL(p_mult · P ‖ Q)`, `Q`
/// the Student-t affinities of `y`, plus `lambda ×` the densMAP loss when
/// `density` is `Some((centered log-target, lambda))`.
///
/// - KL: `4 Σ_j (p_mult·P_ij − S·q_ij) w_ij (y_i − y_j)`, with
///   `w_ij = 1/(1 + d_ij²)`, `q_ij = w_ij / Σ w` and `S = p_mult · Σ P`.
/// - densMAP: `L = mean_i (c_i − t_i)²` with `c` the centered `log R`,
///   `R_i = max(Σ_j P_ij d_ij, 1e-3)`; its gradient is
///   `Σ_j (g_i + g_j) P_ij (y_i − y_j) / d_ij` with `g_k = 2(c_k − t_k) / (n R_k)`
///   (0 where `R` is floored).
fn gradient(
    y: &[f32],
    p: &[f32],
    p_mult: f32,
    p_sum: f32,
    density: Option<(&[f32], f32)>,
) -> Vec<f32> {
    let n = y.len() / 2;
    let pair = |i: usize, j: usize| {
        let (dx, dy) = (y[2 * i] - y[2 * j], y[2 * i + 1] - y[2 * j + 1]);
        (dx, dy, dx * dx + dy * dy)
    };

    // Σ_ij w_ij, and each point's R_i when density is on.
    let rows: Vec<(f64, f32)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let (mut z, mut r) = (0.0_f64, 0.0_f32);
            for j in (0..n).filter(|&j| j != i) {
                let (_, _, d2) = pair(i, j);
                z += f64::from(1.0 / (1.0 + d2));
                if density.is_some() {
                    r += p[i * n + j] * d2.max(1e-12).sqrt();
                }
            }
            (z, r)
        })
        .collect();
    let z = rows.iter().map(|&(z, _)| z).sum::<f64>() as f32;
    let s = p_mult * p_sum;

    // g_k of the density term.
    let g_density: Option<Vec<f32>> = density.map(|(target, lambda)| {
        let log_r: Vec<f32> = rows.iter().map(|&(_, r)| r.max(1e-3).ln()).collect();
        let mean = log_r.iter().sum::<f32>() / n as f32;
        rows.iter()
            .zip(&log_r)
            .zip(target)
            .map(|((&(_, r), &lr), &t)| {
                if r > 1e-3 {
                    lambda * 2.0 * (lr - mean - t) / (n as f32 * r)
                } else {
                    0.0
                }
            })
            .collect()
    });

    let mut grad = vec![0.0_f32; 2 * n];
    grad.par_chunks_exact_mut(2)
        .enumerate()
        .for_each(|(i, gi)| {
            let (mut gx, mut gy) = (0.0_f32, 0.0_f32);
            for j in (0..n).filter(|&j| j != i) {
                let (dx, dy, d2) = pair(i, j);
                let w = 1.0 / (1.0 + d2);
                let pij = p[i * n + j];
                let mut f = 4.0 * (p_mult * pij - s * w / z) * w;
                if let Some(gd) = &g_density {
                    if d2 > 1e-12 {
                        f += (gd[i] + gd[j]) * pij / d2.sqrt();
                    }
                }
                gx += f * dx;
                gy += f * dy;
            }
            gi[0] = gx;
            gi[1] = gy;
        });
    grad
}

/// The joint `P` (row-major `n × n`) of the graph `edges`:
/// `P_ij = (P(j|i) + P(i|j)) / 2n` with `P(j|i) = w_ij / Σ_k w_ik`, floored
/// at 1e-12 for numerical stability.
fn joint_probabilities(n: usize, edges: &[(usize, usize, f32)]) -> Vec<f32> {
    let mut total = vec![0.0_f32; n];
    for &(a, b, w) in edges {
        if a != b {
            total[a] += w;
            total[b] += w;
        }
    }
    let mut p = vec![1e-12_f32; n * n];
    let half = 1.0 / (2.0 * n as f32);
    for &(a, b, w) in edges {
        if a != b {
            let v = half * (w / total[a] + w / total[b]);
            p[a * n + b] += v;
            p[b * n + a] += v;
        }
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    /// t-SNE on a graph's affinities: three groups of 100, each node linked
    /// to its 16 nearest along a ring (as a kNN graph of PBs is), joined by a
    /// few weak links, stay apart.
    #[test]
    fn three_linked_groups_on_a_graph_stay_apart() {
        let (per, n) = (100, 300);
        let mut edges: Vec<(usize, usize, f32)> = Vec::new();
        for g in 0..3 {
            for k in 0..per {
                for s in 1..=8 {
                    edges.push((g * per + k, g * per + (k + s) % per, 4.0));
                }
            }
        }
        for g in 0..3 {
            edges.push((g * per, ((g + 1) % 3) * per + 5, 0.2));
        }
        let args = TsneArgs {
            n_iter: 500,
            seed: 1,
            ..TsneArgs::default()
        };
        let m = tsne_layout_2d_on_graph(n, &edges, &args).unwrap();
        let y: Vec<f32> = (0..n).flat_map(|i| [m[(i, 0)], m[(i, 1)]]).collect();
        let centre = |g: usize| {
            (0..per).fold([0.0_f32; 2], |[a, b], k| {
                let i = g * per + k;
                [a + y[2 * i] / per as f32, b + y[2 * i + 1] / per as f32]
            })
        };
        let c: Vec<[f32; 2]> = (0..3).map(centre).collect();
        let spread = (0..n)
            .map(|i| {
                let g = &c[i / per];
                (y[2 * i] - g[0]).hypot(y[2 * i + 1] - g[1])
            })
            .sum::<f32>()
            / n as f32;
        for a in 0..3 {
            for b in 0..a {
                let apart = (c[a][0] - c[b][0]).hypot(c[a][1] - c[b][1]);
                assert!(
                    apart > 2.0 * spread,
                    "groups {a}, {b}: {apart} apart, spread {spread}"
                );
            }
        }
    }

    /// The candle losses the gradient was written from, for autograd.
    mod reference {
        use candle_core::{DType, Result, Tensor, Var, D};

        fn dist_sq(y: &Var, n: usize) -> Result<Tensor> {
            let y_sq = y.as_tensor().sqr()?.sum(D::Minus1)?;
            let dot = y.as_tensor().matmul(&y.as_tensor().t()?)?;
            y_sq.reshape((n, 1))?
                .broadcast_add(&y_sq.reshape((1, n))?)?
                - (&dot * 2.0)?
        }

        pub fn kl(y: &Var, p: &Tensor, n: usize) -> Result<Tensor> {
            let q = (dist_sq(y, n)? + 1.0)?.recip()?;
            let mask = Tensor::eye(n, DType::F32, y.as_tensor().device())?;
            let q = (&q * (1.0 - &mask)?)?;
            let q = q.broadcast_div(&q.sum_all()?)?.clamp(1e-12, 1.0)?;
            (p * p.broadcast_div(&q)?.log()?)?.sum_all()
        }

        pub fn density(y: &Var, p: &Tensor, target: &Tensor, n: usize) -> Result<Tensor> {
            let dist = dist_sq(y, n)?.clamp(1e-12f32, f32::INFINITY)?.sqrt()?;
            let r = (p * &dist)?.sum(D::Minus1)?;
            let log_r = r.clamp(1e-3f32, f32::INFINITY)?.log()?;
            let c = log_r.broadcast_sub(&log_r.mean_all()?)?;
            (&c - target)?.sqr()?.mean_all()
        }
    }

    /// The written-out gradient matches candle's autograd of the KL and
    /// densMAP losses, with and without exaggeration.
    #[test]
    fn gradient_matches_autograd() {
        use candle_core::{Device, Tensor, Var};
        use rand::{rngs::SmallRng, RngExt, SeedableRng};

        let n = 40;
        let mut rng = SmallRng::seed_from_u64(3);
        let edges: Vec<(usize, usize, f32)> = (0..n)
            .flat_map(|i| (1..=4).map(move |s| (i, (i + s * 3) % n)))
            .map(|(a, b)| (a, b, rng.random_range(0.5..2.0)))
            .collect();
        let y: Vec<f32> = (0..2 * n).map(|_| rng.random_range(-3.0..3.0)).collect();
        let mut target: Vec<f32> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();
        let mean = target.iter().sum::<f32>() / n as f32;
        target.iter_mut().for_each(|t| *t -= mean);

        let p = joint_probabilities(n, &edges);
        let p_sum: f32 = p.iter().sum();
        let dev = Device::Cpu;
        let pt = Tensor::from_vec(p.clone(), (n, n), &dev).unwrap();
        let tt = Tensor::from_vec(target.clone(), (n,), &dev).unwrap();
        let lambda = 0.7_f32;

        for (p_mult, dens) in [(12.0_f32, false), (1.0, false), (1.0, true)] {
            let yv = Var::from_tensor(&Tensor::from_vec(y.clone(), (n, 2), &dev).unwrap()).unwrap();
            let mut loss = reference::kl(&yv, &(&pt * f64::from(p_mult)).unwrap(), n).unwrap();
            if dens {
                let d = reference::density(&yv, &pt, &tt, n).unwrap();
                loss = (loss + (d * f64::from(lambda)).unwrap()).unwrap();
            }
            let auto: Vec<f32> = loss
                .backward()
                .unwrap()
                .get(&yv)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let ours = gradient(&y, &p, p_mult, p_sum, dens.then_some((&target[..], lambda)));
            let scale = auto.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
            for (a, o) in auto.iter().zip(&ours) {
                assert!(
                    (a - o).abs() <= 1e-3 * scale,
                    "p_mult {p_mult}, density {dens}: autograd {a}, ours {o} (max {scale})"
                );
            }
        }
    }
}
