//! Gaussian mixtures with diagonal covariances on the rows of a dense
//! matrix, fitted by EM from a seeded k-means start, and the number of
//! components chosen by BIC on the (soft) log-likelihood.
//!
//! A BIC read off hard k-means assignments always rewards cutting a Gaussian
//! in two (each half is narrower than the whole); under a mixture's soft
//! likelihood two overlapping halves explain the data no better than one,
//! so the count of components follows the data's clumps, not its spread.
//!
//! Each component has a weight, a mean and a variance per dimension
//! (floored at `VAR_FLOOR` times that dimension's overall variance, so a
//! component cannot collapse on a point). E-steps run over fixed row chunks
//! and sum in chunk order, so a fit depends on the seed and the data, never
//! on the thread count.

use crate::matrix::kmeans::{kmeans_rows_best, KmeansRowsOpts};
use nalgebra::DMatrix;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Rows per E-step task (fixed, so sums do not depend on the thread count).
const CHUNK_ROWS: usize = 4096;
/// A component's variance floor, as a share of the dimension's variance.
pub const VAR_FLOOR: f64 = 1e-3;
/// EM stops when the mean log-likelihood per row gains less than this.
pub const TOL: f64 = 1e-7;

/// A fitted mixture: `k` components over `d` dimensions.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DiagGmm {
    pub weights: Vec<f64>,
    /// `k` rows of `d` means, and of `d` variances.
    pub means: Vec<Vec<f32>>,
    pub vars: Vec<Vec<f32>>,
}

/// A mixture with each row's most probable component, the log-likelihood
/// of the rows and the entropy of their component memberships (`−Σ r ln r`).
pub struct GmmFit {
    pub model: DiagGmm,
    pub labels: Vec<usize>,
    pub log_lik: f64,
    pub entropy: f64,
}

/// What chooses the number of components.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Criterion {
    /// `−2 log L + p ln n`: the density's fit.
    Bic,
    /// BIC plus twice the memberships' entropy (Biernacki, Celeux & Govaert
    /// 2000): components that overlap, as two Gaussians fitting one skewed
    /// hump do, are charged for it, so the count follows separated groups.
    Icl,
}

impl GmmFit {
    /// The fit's score under `criterion` over its `n` rows; lower is better.
    #[must_use]
    pub fn score(&self, criterion: Criterion, n: usize) -> f64 {
        let bic = self.model.bic(self.log_lik, n);
        match criterion {
            Criterion::Bic => bic,
            Criterion::Icl => bic + 2.0 * self.entropy,
        }
    }
}

impl DiagGmm {
    #[must_use]
    pub fn k(&self) -> usize {
        self.weights.len()
    }

    /// Free parameters: `k − 1` weights, `k·d` means and `k·d` variances.
    #[must_use]
    pub fn n_params(&self) -> usize {
        let d = self.means.first().map_or(0, Vec::len);
        self.k() - 1 + 2 * self.k() * d
    }

    /// `−2 log L + p ln n` for a log-likelihood over `n` rows; lower is better.
    #[must_use]
    pub fn bic(&self, log_lik: f64, n: usize) -> f64 {
        -2.0 * log_lik + self.n_params() as f64 * (n.max(1) as f64).ln()
    }

    /// Each row's log-density under each component plus its log-weight.
    fn log_joint(&self, row: &[f32], out: &mut [f64]) {
        for (c, o) in out.iter_mut().enumerate() {
            let mut s = self.weights[c].max(f64::MIN_POSITIVE).ln();
            for (j, &x) in row.iter().enumerate() {
                let var = f64::from(self.vars[c][j]);
                let dx = f64::from(x) - f64::from(self.means[c][j]);
                s -= 0.5 * ((2.0 * std::f64::consts::PI * var).ln() + dx * dx / var);
            }
            *o = s;
        }
    }

    /// Each row of `z`'s most probable component (ties to the lowest).
    #[must_use]
    pub fn assign(&self, z: &DMatrix<f32>) -> Vec<usize> {
        let rows = row_major(z);
        let d = z.ncols();
        rows.par_chunks(CHUNK_ROWS * d.max(1))
            .flat_map_iter(|chunk| {
                let mut lj = vec![0f64; self.k()];
                chunk
                    .chunks(d.max(1))
                    .map(|row| {
                        self.log_joint(row, &mut lj);
                        argmax(&lj)
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }
}

/// The transpose's column-major storage: `z` row-major, each row a slice.
fn row_major(z: &DMatrix<f32>) -> Vec<f32> {
    z.transpose().as_slice().to_vec()
}

fn argmax(v: &[f64]) -> usize {
    let mut best = 0;
    for (c, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = c;
        }
    }
    best
}

fn log_sum_exp(v: &[f64]) -> f64 {
    let m = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        return m;
    }
    m + v.iter().map(|&x| (x - m).exp()).sum::<f64>().ln()
}

/// Sufficient statistics of one E-step over a chunk of rows.
struct Stats {
    n: Vec<f64>,
    sx: Vec<f64>,
    sxx: Vec<f64>,
    log_lik: f64,
}

impl Stats {
    fn zeros(k: usize, d: usize) -> Self {
        Self {
            n: vec![0.0; k],
            sx: vec![0.0; k * d],
            sxx: vec![0.0; k * d],
            log_lik: 0.0,
        }
    }

    fn add(&mut self, o: &Self) {
        for (a, b) in [
            (&mut self.n, &o.n),
            (&mut self.sx, &o.sx),
            (&mut self.sxx, &o.sxx),
        ] {
            a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
        }
        self.log_lik += o.log_lik;
    }
}

/// `z` as EM reads it: row-major, and each dimension's variance floor.
struct Rows {
    data: Vec<f32>,
    n: usize,
    d: usize,
    floors: Vec<f64>,
}

impl Rows {
    fn new(z: &DMatrix<f32>) -> Self {
        let (n, d) = (z.nrows(), z.ncols());
        let floors = (0..d)
            .map(|j| {
                let col = z.column(j);
                let m = col.iter().map(|&x| f64::from(x)).sum::<f64>() / n.max(1) as f64;
                let v =
                    col.iter().map(|&x| (f64::from(x) - m).powi(2)).sum::<f64>() / n.max(1) as f64;
                (VAR_FLOOR * v).max(1e-12)
            })
            .collect();
        Self {
            data: row_major(z),
            n,
            d,
            floors,
        }
    }
}

/// EM on the rows of `z` (`n × d`) from `init`'s components, for at most
/// `max_iter` steps; components left with no weight are dropped.
#[must_use]
pub fn fit_diag_gmm(z: &DMatrix<f32>, init: &DiagGmm, max_iter: usize) -> GmmFit {
    fit_rows(&Rows::new(z), init, max_iter)
}

fn fit_rows(z: &Rows, init: &DiagGmm, max_iter: usize) -> GmmFit {
    let (n, d, rows, floors) = (z.n, z.d, &z.data, &z.floors);
    let mut model = init.clone();
    let mut last = f64::NEG_INFINITY;
    for _ in 0..max_iter.max(1) {
        let k = model.k();
        let stats = rows
            .par_chunks(CHUNK_ROWS * d.max(1))
            .map(|chunk| {
                let mut s = Stats::zeros(k, d);
                let mut lj = vec![0f64; k];
                for row in chunk.chunks(d.max(1)) {
                    model.log_joint(row, &mut lj);
                    let lse = log_sum_exp(&lj);
                    s.log_lik += lse;
                    for (c, &l) in lj.iter().enumerate() {
                        let r = (l - lse).exp();
                        s.n[c] += r;
                        for (j, &x) in row.iter().enumerate() {
                            let x = f64::from(x);
                            s.sx[c * d + j] += r * x;
                            s.sxx[c * d + j] += r * x * x;
                        }
                    }
                }
                s
            })
            .collect::<Vec<_>>()
            .into_iter()
            .fold(Stats::zeros(k, d), |mut a, b| {
                a.add(&b);
                a
            });
        // M-step, dropping components with no weight left.
        let keep: Vec<usize> = (0..k).filter(|&c| stats.n[c] > 1e-8).collect();
        model = DiagGmm {
            weights: keep.iter().map(|&c| stats.n[c] / n as f64).collect(),
            means: keep
                .iter()
                .map(|&c| {
                    (0..d)
                        .map(|j| (stats.sx[c * d + j] / stats.n[c]) as f32)
                        .collect()
                })
                .collect(),
            vars: keep
                .iter()
                .map(|&c| {
                    (0..d)
                        .map(|j| {
                            let mu = stats.sx[c * d + j] / stats.n[c];
                            let v = stats.sxx[c * d + j] / stats.n[c] - mu * mu;
                            v.max(floors[j]) as f32
                        })
                        .collect()
                })
                .collect(),
        };
        let gain = (stats.log_lik - last) / n.max(1) as f64;
        last = stats.log_lik;
        if gain.abs() < TOL {
            break;
        }
    }
    // Each row's label, and the log-likelihood and memberships' entropy,
    // under the final parameters, in one pass.
    let k = model.k();
    let parts: Vec<(Vec<usize>, f64, f64)> = rows
        .par_chunks(CHUNK_ROWS * d.max(1))
        .map(|chunk| {
            let mut lj = vec![0f64; k];
            let mut labels = Vec::with_capacity(chunk.len() / d.max(1));
            let (mut ll, mut h) = (0f64, 0f64);
            for row in chunk.chunks(d.max(1)) {
                model.log_joint(row, &mut lj);
                labels.push(argmax(&lj));
                let lse = log_sum_exp(&lj);
                ll += lse;
                h += lj
                    .iter()
                    .map(|&x| {
                        let r = (x - lse).exp();
                        if r > 0.0 {
                            -r * (x - lse)
                        } else {
                            0.0
                        }
                    })
                    .sum::<f64>();
            }
            (labels, ll, h)
        })
        .collect();
    let (mut labels, mut log_lik, mut entropy) = (Vec::with_capacity(n), 0f64, 0f64);
    for (l, ll, h) in parts {
        labels.extend(l);
        log_lik += ll;
        entropy += h;
    }
    GmmFit {
        model,
        labels,
        log_lik,
        entropy,
    }
}

/// A `k`-component start: the best of `n_starts` seeded k-means (seeds
/// `seed + s`), each cluster's weight, mean and variance.
#[must_use]
pub fn kmeans_start(z: &DMatrix<f32>, k: usize, n_starts: usize, seed: u64) -> DiagGmm {
    let (n, d) = (z.nrows(), z.ncols());
    let fit = kmeans_rows_best(z, &KmeansRowsOpts::euclidean(k.max(1), 100, seed), n_starts);
    let kk = fit.centroids.nrows();
    let mut count = vec![0usize; kk];
    let mut ss = vec![vec![0f64; d]; kk];
    for (i, &c) in fit.labels.iter().enumerate() {
        count[c] += 1;
        for j in 0..d {
            ss[c][j] += f64::from(z[(i, j)] - fit.centroids[(c, j)]).powi(2);
        }
    }
    DiagGmm {
        weights: count
            .iter()
            .map(|&m| (m.max(1) as f64) / n.max(1) as f64)
            .collect(),
        means: (0..kk)
            .map(|c| fit.centroids.row(c).iter().copied().collect())
            .collect(),
        vars: (0..kk)
            .map(|c| {
                (0..d)
                    .map(|j| (ss[c][j] / count[c].max(1) as f64).max(1e-6) as f32)
                    .collect()
            })
            .collect(),
    }
}

/// `prev` with its widest component along dimension `j` (largest weight
/// times variance there) split in two at its mean ± one SD: a start that a
/// k-means start can miss, when a noisier dimension draws its cut.
#[must_use]
pub fn split_start(prev: &DiagGmm, j: usize) -> DiagGmm {
    let c = (0..prev.k())
        .max_by(|&a, &b| {
            let wa = prev.weights[a] * f64::from(prev.vars[a][j]);
            let wb = prev.weights[b] * f64::from(prev.vars[b][j]);
            wa.total_cmp(&wb)
        })
        .expect("a component");
    let sd = prev.vars[c][j].sqrt();
    let mut next = prev.clone();
    next.weights[c] /= 2.0;
    next.weights.push(next.weights[c]);
    let mut other = prev.means[c].clone();
    next.means[c][j] -= sd;
    other[j] += sd;
    next.means.push(other);
    next.vars.push(prev.vars[c].clone());
    next
}

/// Rows a choice of `k` is made on: a seeded subsample of this many when
/// there are more; the chosen mixture is then refitted on every row.
pub const SELECT_ROWS: usize = 50_000;
/// EM steps for each candidate start before the best is run to convergence.
const CANDIDATE_STEPS: usize = 25;

/// The mixture of each `k` in `1..=k_max` (at most the number of rows) and
/// the one of least `criterion` ([`Criterion`]); a tie goes to the smaller
/// `k`. Each `k` tries the best of `n_starts` seeded k-means
/// ([`kmeans_start`]) and the previous `k`'s fit split along each dimension
/// ([`split_start`]), a few EM steps each, and runs the likeliest on. `k` is
/// chosen on at most [`SELECT_ROWS`] rows (a seeded subsample).
/// `score[k − 1]` is each `k`'s criterion on those rows.
pub struct GmmChoice {
    /// The chosen mixture, fitted on the rows `k` was chosen on (all of
    /// them after [`GmmChoice::on_all`] or from [`choose_gmm`]).
    pub fit: GmmFit,
    pub score: Vec<f64>,
    /// Whether `fit` is on a subsample.
    pub subsampled: bool,
}

impl GmmChoice {
    /// The choice refitted on every row of `z` when it was made on a
    /// subsample.
    #[must_use]
    pub fn on_all(self, z: &DMatrix<f32>, max_iter: usize) -> Self {
        if !self.subsampled {
            return self;
        }
        Self {
            fit: fit_diag_gmm(z, &self.fit.model, max_iter),
            score: self.score,
            subsampled: false,
        }
    }
}

/// See [`GmmChoice`]; the chosen mixture refitted on every row.
#[must_use]
pub fn choose_gmm(
    z: &DMatrix<f32>,
    k_max: usize,
    criterion: Criterion,
    n_starts: usize,
    seed: u64,
    max_iter: usize,
) -> GmmChoice {
    select_gmm(z, k_max, criterion, n_starts, seed, max_iter).on_all(z, max_iter)
}

/// See [`GmmChoice`]; the chosen mixture as fitted on the rows `k` was
/// chosen on, for a caller that needs only `k` or refits later.
#[must_use]
pub fn select_gmm(
    z: &DMatrix<f32>,
    k_max: usize,
    criterion: Criterion,
    n_starts: usize,
    seed: u64,
    max_iter: usize,
) -> GmmChoice {
    let n = z.nrows();
    let k_max = k_max.min(n).max(1);
    let sub: Option<DMatrix<f32>> = (n > SELECT_ROWS).then(|| {
        let mut rng = <rand::rngs::SmallRng as rand::SeedableRng>::seed_from_u64(seed);
        let mut ids = rand::seq::index::sample(&mut rng, n, SELECT_ROWS).into_vec();
        ids.sort_unstable();
        z.select_rows(&ids)
    });
    let zs = sub.as_ref().unwrap_or(z);
    let rows = Rows::new(zs);
    let mut score = Vec::with_capacity(k_max);
    let mut best: Option<(f64, GmmFit)> = None;
    let mut prev: Option<DiagGmm> = None;
    for k in 1..=k_max {
        let mut starts = vec![kmeans_start(zs, k, n_starts, seed)];
        if let Some(p) = &prev {
            starts.extend((0..zs.ncols()).map(|j| split_start(p, j)));
        }
        // Seeded starts in parallel; the likeliest taken in start order.
        let likeliest = starts
            .par_iter()
            .map(|s| fit_rows(&rows, s, CANDIDATE_STEPS.min(max_iter)))
            .collect::<Vec<_>>()
            .into_iter()
            .max_by(|a, b| a.log_lik.total_cmp(&b.log_lik))
            .expect("a start");
        let fit = fit_rows(&rows, &likeliest.model, max_iter);
        let b = fit.score(criterion, rows.n);
        score.push(b);
        prev = Some(fit.model.clone());
        if best.as_ref().is_none_or(|(least, _)| b < *least) {
            best = Some((b, fit));
        }
    }
    GmmChoice {
        fit: best.expect("k = 1 at least").1,
        score,
        subsampled: sub.is_some(),
    }
}

#[cfg(test)]
#[path = "gmm_tests.rs"]
mod tests;
