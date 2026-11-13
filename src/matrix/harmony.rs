//! Harmony (Korsunsky et al. 2019, "Fast, sensitive and accurate
//! integration of single-cell data with Harmony", Nature Methods
//! 16:1289-1296, doi:10.1038/s41592-019-0619-0) on a PCA embedding, with
//! one batch variable.
//!
//! Section and equation numbers are those of the paper's Online Methods;
//! iteration counts and tolerances are those of its reference
//! implementations (R `harmony`, `harmonypy`).
//!
//! - Clustering runs on the cells L2-normalised, where the squared distance
//!   to a unit centroid is `2 (1 - cosine)` (section 2.2, eq. 4).
//! - Centroids start from k-means on the normalised cells, the best of 10
//!   k-means++ starts, then L2-normalised (section 2.6.2); the starting `R`
//!   is the soft assignment `exp(-d / sigma)`, each cell's column summed to 1.
//! - Maximum diversity clustering (section 2) minimises eq. 3: the soft
//!   k-means distance, `sigma` times the entropy term `sum R ln R`, and the
//!   diversity penalty `sigma theta sum R ln((O + 1) / (E + 1))` over each
//!   cell's batch, with 1 added to `O` and `E` (section 2.6.3). Each round
//!   sets the centroids to the normalised `R`-weighted sum of the cells
//!   (Dhillon's spherical update, section 2.4.2), then `R` by eq. 8,
//!   `R_ki ∝ exp(-d_ki / sigma) ((E_kb + 1) / (O_kb + 1))^theta`, in blocks of
//!   5% of the cells in random order, `O` and `E` counted without the block
//!   (section 2.6.1). Rounds stop when the sum of the last 3 objectives moves
//!   by less than 1e-5 of the previous 3, or after 20 rounds.
//! - The correction (section 3, eqs. 9 to 13) fits, per cluster `k`, the
//!   ridge mixture of experts `W_k = (Φ* diag(R_k) Φ*ᵀ + Λ)⁻¹ Φ* diag(R_k) Zᵀ`,
//!   `Φ*` the batch one-hot design with an intercept row and `Λ =
//!   diag(0, lambda, ...)`, and subtracts the batch rows only:
//!   `Ẑ = Z - Σ_k W_k[1..]ᵀ Φ diag(R_k)`, always from the original `Z`
//!   (section 1.1). Clustering then resumes on `Ẑ` normalised.
//! - Harmony rounds stop when the objective falls by less than 1e-4 of its
//!   value, or after `max_rounds` rounds.
//!
//! Eq. 8 is evaluated in log space and shifted by each cell's maximum
//! before `exp`, so no setting overflows or underflows a whole column.
//! Batch labels are any `usize`s; they are renumbered `0..B` first.
//!
//! Simplification: theta is not discounted for small batches (section
//! 2.6.4 with tau = 0, the reference implementations' default).

use crate::matrix::kmeans::kmeans_centroids_seeded;
use crate::matrix::traits::MatOps;
use nalgebra::{DMatrix, DVector, DVectorView};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;

const MAX_CLUSTER_ROUNDS: usize = 20;
const BLOCK_FRAC: f64 = 0.05;
const EPS_CLUSTER: f64 = 1e-5;
const EPS_HARMONY: f64 = 1e-4;
const WINDOW: usize = 3;
const KMEANS_STARTS: u64 = 10;
const KMEANS_ROUNDS: usize = 25;
/// Cells per rayon task when assigning a block; smaller blocks stay on one thread.
const MIN_PAR_CELLS: usize = 256;
/// Entries per serial partial sum in the objective, so its parallel sum
/// is added in the same order every run.
const SUM_CHUNK: usize = 1 << 14;

/// Harmony's settings; the defaults are the reference implementations'.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HarmonyArgs {
    /// Diversity penalty `theta` (>= 0)
    pub theta: f64,
    /// Soft-clustering width `sigma` (> 0)
    pub sigma: f64,
    /// Ridge penalty `lambda` (> 0)
    pub lambda: f64,
    /// Clusters; `None` is `min(100, N / 30)`
    pub n_clusters: Option<usize>,
    /// Harmony rounds at most (>= 1)
    pub max_rounds: usize,
}

impl Default for HarmonyArgs {
    fn default() -> Self {
        Self {
            theta: 2.0,
            sigma: 0.1,
            lambda: 1.0,
            n_clusters: None,
            max_rounds: 10,
        }
    }
}

impl HarmonyArgs {
    /// Settings Harmony can correct with: no ridge penalty leaves every
    /// cluster's system singular, and no width every assignment NaN.
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.theta >= 0.0, "theta must be at least 0");
        anyhow::ensure!(self.sigma > 0.0, "sigma must be positive");
        anyhow::ensure!(self.lambda > 0.0, "lambda must be positive");
        anyhow::ensure!(self.max_rounds >= 1, "max_rounds must be at least 1");
        anyhow::ensure!(self.n_clusters != Some(0), "n_clusters must be at least 1");
        Ok(())
    }
}

/// The mixture-of-experts ridge correction (section 3) of `z` (dims ×
/// cells, `zt` its transpose) under soft clusters `r` (clusters × cells),
/// with `cells[b]` the cells of batch `b`. Per batch, `R_b Z_bᵀ` gives
/// every cluster's right-hand side at once and `W_bᵀ R_b` its whole
/// correction.
fn correct(
    z: &DMatrix<f64>,
    zt: &DMatrix<f64>,
    r: &DMatrix<f64>,
    cells: &[Vec<usize>],
    lambda: f64,
) -> DMatrix<f64> {
    let (n_clusters, d, n_batches) = (r.nrows(), z.nrows(), cells.len());
    let r_b: Vec<DMatrix<f64>> = cells.iter().map(|c| r.select_columns(c)).collect();
    // clusters × dims: Σ_{i in b} R_ki z_i
    let m_b: Vec<DMatrix<f64>> = cells
        .iter()
        .zip(&r_b)
        .map(|(c, rb)| rb * zt.select_rows(c))
        .collect();
    // O_kb, clusters × batches
    let o = DMatrix::from_columns(&r_b.iter().map(|rb| rb.column_sum()).collect::<Vec<_>>());

    // row k of w_b[b] is W_k's batch-b row
    let mut w_b = vec![DMatrix::<f64>::zeros(n_clusters, d); n_batches];
    for k in 0..n_clusters {
        // Φ* diag(R_k) Φ*ᵀ + Λ and Φ* diag(R_k) Zᵀ; row 0 is the intercept
        let mut a = DMatrix::<f64>::zeros(n_batches + 1, n_batches + 1);
        let mut rhs = DMatrix::<f64>::zeros(n_batches + 1, d);
        for (b, mb) in m_b.iter().enumerate() {
            let (okb, j) = (o[(k, b)], b + 1);
            a[(0, 0)] += okb;
            a[(0, j)] = okb;
            a[(j, 0)] = okb;
            a[(j, j)] = okb + lambda;
            let row = mb.row(k);
            rhs.set_row(j, &row);
            let mut r0 = rhs.row_mut(0);
            r0 += &row;
        }
        // singular only for a cluster holding no weight, which moves no cell
        let Some(w) = a.lu().solve(&rhs) else {
            continue;
        };
        for (b, wb) in w_b.iter_mut().enumerate() {
            wb.set_row(k, &w.row(b + 1));
        }
    }

    let mut out = z.clone();
    for ((c, rb), wb) in cells.iter().zip(&r_b).zip(&w_b) {
        let delta = wb.tr_mul(rb);
        for (t, &i) in c.iter().enumerate() {
            let mut col = out.column_mut(i);
            col -= delta.column(t);
        }
    }
    out
}

/// Eq. 8 for one cell into `out`, in log space: `-d_k / sigma +
/// log_w(k)`, shifted by its maximum before `exp`, then summed to 1.
fn soft_assign(dist: DVectorView<f64>, sigma: f64, log_w: impl Fn(usize) -> f64, out: &mut [f64]) {
    for (k, v) in out.iter_mut().enumerate() {
        *v = -dist[k] / sigma + log_w(k);
    }
    let max = out.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    out.iter_mut().for_each(|v| *v = (*v - max).exp());
    let total: f64 = out.iter().sum();
    out.iter_mut().for_each(|v| *v /= total);
}

/// Centroids (dims × clusters, unit columns) from k-means on the cells of
/// `z_cos` (dims × cells): the best of [`KMEANS_STARTS`] seeds (one when
/// `k` is 1, where every start gives the mean).
fn initial_centroids(z_cos: &DMatrix<f64>, k: usize, seed: u64) -> DMatrix<f64> {
    let rows = z_cos.transpose().map(|v| v as f32);
    let cols = z_cos.map(|v| v as f32);
    let starts = if k > 1 { KMEANS_STARTS } else { 1 };
    let (_, cents) = (0..starts)
        .map(|s| {
            let (cents, labels) =
                kmeans_centroids_seeded(&rows, k, KMEANS_ROUNDS, seed.wrapping_add(s));
            let cents_t = cents.transpose();
            let per_cell: Vec<f64> = labels
                .par_iter()
                .enumerate()
                .map(|(i, &l)| {
                    let sq: f32 = cols
                        .column(i)
                        .iter()
                        .zip(cents_t.column(l).iter())
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum();
                    sq as f64
                })
                .collect();
            let inertia: f64 = per_cell.iter().sum();
            (inertia, cents)
        })
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .expect("at least one k-means start");
    let mut y = cents.transpose().map(|v| v as f64);
    y.normalize_columns_inplace();
    y
}

/// Maximum diversity clustering's state (section 2).
struct Clustering<'a> {
    /// Batch of each cell, in `0..B`.
    batch: &'a [usize],
    args: &'a HarmonyArgs,
    /// Each batch's share of the cells.
    pr_b: Vec<f64>,
    /// Centroids, dims × clusters.
    y: DMatrix<f64>,
    /// Soft assignments, clusters × cells.
    r: DMatrix<f64>,
    /// Observed cells per cluster and batch, clusters × batches.
    o: DMatrix<f64>,
    /// Cells per cluster, `Σ_i R_ki`; the expected count is `E_kb = mass_k pr_b`.
    mass: DVector<f64>,
    /// Every objective so far, over all Harmony rounds.
    objectives: Vec<f64>,
    rng: SmallRng,
}

impl Clustering<'_> {
    /// Squared distances `2 (1 - yᵀ z)` into `dist`, clusters × cells.
    fn distances(&self, z_cos: &DMatrix<f64>, dist: &mut DMatrix<f64>) {
        self.y.tr_mul_to(z_cos, dist);
        dist.as_mut_slice()
            .par_iter_mut()
            .for_each(|c| *c = 2.0 * (1.0 - *c));
    }

    /// `ln((O_kb + 1) / (E_kb + 1))`, clusters × batches.
    fn log_ratio(&self) -> DMatrix<f64> {
        DMatrix::from_fn(self.o.nrows(), self.o.ncols(), |k, b| {
            ((self.o[(k, b)] + 1.0) / (self.mass[k] * self.pr_b[b] + 1.0)).ln()
        })
    }

    /// `±` cell `i`'s assignment in `O` and the cluster masses.
    fn tally(&mut self, i: usize, sign: f64) {
        let b = self.batch[i];
        for k in 0..self.r.nrows() {
            let v = sign * self.r[(k, i)];
            self.o[(k, b)] += v;
            self.mass[k] += v;
        }
    }

    /// Eq. 3; the diversity term summed per cluster and batch,
    /// `Σ_kb O_kb ln((O_kb + 1) / (E_kb + 1))`.
    fn objective(&self, dist: &DMatrix<f64>) -> f64 {
        let HarmonyArgs { sigma, theta, .. } = *self.args;
        let partial: Vec<f64> = self
            .r
            .as_slice()
            .par_chunks(SUM_CHUNK)
            .zip(dist.as_slice().par_chunks(SUM_CHUNK))
            .map(|(r, d)| {
                r.iter()
                    .zip(d)
                    .filter(|(&r, _)| r > 0.0)
                    .map(|(&r, &d)| r * d + sigma * r * r.ln())
                    .sum()
            })
            .collect();
        let fit: f64 = partial.iter().sum();
        let diversity: f64 = self
            .o
            .iter()
            .zip(self.log_ratio().iter())
            .map(|(o, l)| o * l)
            .sum();
        fit + sigma * theta * diversity
    }

    /// Eq. 8, in shuffled blocks (section 2.6.1); the cells of a block see
    /// the same `O` and `E`, so they are assigned in parallel.
    fn update_r(&mut self, dist: &DMatrix<f64>) {
        let n = self.batch.len();
        let HarmonyArgs { sigma, theta, .. } = *self.args;
        let mut order: Vec<usize> = (0..n).collect();
        order.shuffle(&mut self.rng);
        let block = ((n as f64 * BLOCK_FRAC).ceil() as usize).max(1);
        let k = self.r.nrows();
        let mut buf = vec![0.0; k * block];
        for cells in order.chunks(block) {
            cells.iter().for_each(|&i| self.tally(i, -1.0));
            let lr = self.log_ratio();
            let batch = self.batch;
            let buf = &mut buf[..k * cells.len()];
            buf.par_chunks_mut(k)
                .with_min_len(MIN_PAR_CELLS)
                .zip(cells.par_iter())
                .for_each(|(out, &i)| {
                    soft_assign(dist.column(i), sigma, |j| -theta * lr[(j, batch[i])], out)
                });
            for (out, &i) in buf.chunks(k).zip(cells) {
                self.r.column_mut(i).copy_from_slice(out);
                self.tally(i, 1.0);
            }
        }
    }

    /// Clustering rounds on `z_cos` until the windowed objective settles;
    /// `dist` is scratch space, clusters × cells.
    fn cluster(&mut self, z_cos: &DMatrix<f64>, dist: &mut DMatrix<f64>) {
        let z_cos_t = z_cos.transpose();
        for round in 0..MAX_CLUSTER_ROUNDS {
            self.y = (&self.r * &z_cos_t).transpose();
            self.y.normalize_columns_inplace();
            self.distances(z_cos, dist);
            let dist = &*dist;
            self.update_r(dist);
            self.objectives.push(self.objective(dist));
            if round > WINDOW && self.settled() {
                break;
            }
        }
    }

    fn settled(&self) -> bool {
        let m = self.objectives.len();
        let old: f64 = self.objectives[m - 1 - WINDOW..m - 1].iter().sum();
        let new: f64 = self.objectives[m - WINDOW..].iter().sum();
        (old - new).abs() < EPS_CLUSTER * old.abs()
    }
}

/// The Harmony-corrected embedding of `x` (cells × PCs), cell `i` in batch
/// `batch[i]`; `seed` drives the k-means starts and the block order.
pub fn harmonize(
    x: &DMatrix<f32>,
    batch: &[usize],
    p: &HarmonyArgs,
    seed: u64,
) -> anyhow::Result<DMatrix<f32>> {
    p.validate()?;
    let n = x.nrows();
    anyhow::ensure!(n > 0, "no cells");
    anyhow::ensure!(
        x.iter().all(|v| v.is_finite()),
        "non-finite embedding entry"
    );
    anyhow::ensure!(
        batch.len() == n,
        "{} batch labels for {n} cells",
        batch.len()
    );
    let mut labels = batch.to_vec();
    labels.sort_unstable();
    labels.dedup();
    let batch: Vec<usize> = batch
        .iter()
        .map(|b| labels.binary_search(b).expect("label listed"))
        .collect();
    let n_batches = labels.len();
    let mut cells = vec![Vec::new(); n_batches];
    for (i, &b) in batch.iter().enumerate() {
        cells[b].push(i);
    }
    let pr_b: Vec<f64> = cells.iter().map(|c| c.len() as f64 / n as f64).collect();

    let k = match p.n_clusters {
        Some(k) => {
            anyhow::ensure!(k <= n, "{k} clusters for {n} cells");
            k
        }
        None => ((n as f64 / 30.0).round() as usize).clamp(1, 100.min(n)),
    };
    let zt = x.map(|v| v as f64);
    let z = zt.transpose();
    let mut z_cos = z.normalize_columns();

    let mut c = Clustering {
        batch: &batch,
        args: p,
        pr_b,
        y: initial_centroids(&z_cos, k, seed),
        r: DMatrix::zeros(k, n),
        o: DMatrix::zeros(k, n_batches),
        mass: DVector::zeros(k),
        objectives: Vec::new(),
        rng: SmallRng::seed_from_u64(seed),
    };
    let mut dist = DMatrix::zeros(k, n);
    c.distances(&z_cos, &mut dist);
    c.r.as_mut_slice()
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(i, out)| soft_assign(dist.column(i), p.sigma, |_| 0.0, out));
    (0..n).for_each(|i| c.tally(i, 1.0));
    c.objectives.push(c.objective(&dist));

    let mut round = 0;
    let z_hat = loop {
        let old = *c.objectives.last().expect("an objective per round");
        c.cluster(&z_cos, &mut dist);
        let z_hat = correct(&z, &zt, &c.r, &cells, p.lambda);
        round += 1;
        let new = *c.objectives.last().expect("an objective per round");
        if round == p.max_rounds || old - new < EPS_HARMONY * old.abs() {
            break z_hat;
        }
        z_cos = z_hat;
        z_cos.normalize_columns_inplace();
    };
    Ok(DMatrix::from_fn(n, z_hat.nrows(), |i, c| {
        z_hat[(c, i)] as f32
    }))
}

#[cfg(test)]
mod tests;
