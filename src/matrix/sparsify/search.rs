//! DSE on one connected piece: the search of Fitz-Gibbon et al. (2026,
//! Algorithm 1), each step ranked by Φ's exact gradient ([`Gradient`]):
//!
//! 1. the maximum spanning tree, so the sparse graph is connected;
//! 2. edges added in batches, those whose weight lowers Φ most first, to
//!    the budget;
//! 3. every weight rescaled by `(n − 1) / Σ w̃ R` (`R` the full graph's
//!    effective resistance), kept when it lowers Φ;
//! 4. swaps of a kept edge for one left out, ranked over the 20 best of
//!    each, a few checked exactly, the best that lowers Φ taken, and none
//!    that disconnects;
//! 5. the kept weights fitted by Adam on their logs, each at most 20 times
//!    the largest full weight;
//!
//! 4 and 5 done `swap_rounds` times. Each step costs one eigendecomposition
//! of the sparse Laplacian, O(n³); a spectrum already in hand (a swap
//! checked, a rescale, the best fit) is passed on, not computed again.

use super::spectral::{phi, Edge, Gradient, Spectrum};
use super::DseArgs;
use crate::matrix::graph::{is_connected, maximum_spanning_tree};

/// Candidates of each kind a swap is chosen among.
const SWAP_CANDIDATES: usize = 20;

/// A weight's ceiling, as a multiple of the largest full weight.
const MAX_WEIGHT_GAIN: f64 = 20.0;

/// Adam's moment decays.
const ADAM_BETAS: (f64, f64) = (0.9, 0.999);

/// A sparse graph — a weight per full edge, 0 for one left out — and its
/// spectrum.
struct Sparse {
    w: Vec<f64>,
    spectrum: Spectrum,
}

/// One connected piece and the search over its sparse subgraphs.
pub(super) struct Piece<'a> {
    n: usize,
    edges: &'a [Edge],
    full: Spectrum,
    args: &'a DseArgs,
}

impl<'a> Piece<'a> {
    /// The connected `n`-node graph `edges` (each pair once).
    pub fn new(n: usize, edges: &'a [Edge], args: &'a DseArgs) -> Self {
        Self {
            n,
            edges,
            full: Spectrum::of(n, edges),
            args,
        }
    }

    /// The sparse graph of `keep` edges (at least a spanning tree's) and its
    /// Φ: a weight per full edge, 0 where left out, each kept weight as it
    /// will be written (f32), and Φ of those.
    pub fn sparsify(&self, keep: usize) -> (Vec<f64>, f64) {
        let mut w = self.spanning_tree();
        self.add_to(&mut w, keep);
        let mut sparse = self.rescaled(w);
        for _ in 0..self.args.swap_rounds {
            let mut g = Gradient::new(&self.full, &sparse.spectrum);
            // Without an exact check no swap can be taken.
            let swaps = if self.args.exact_checks == 0 {
                0
            } else {
                self.args.max_swaps
            };
            for _ in 0..swaps {
                match self.swap(&sparse.w, &g) {
                    Some(next) => {
                        sparse = next;
                        g = Gradient::new(&self.full, &sparse.spectrum);
                    }
                    None => break,
                }
            }
            sparse = self.fit(sparse, g);
        }
        let mut w = sparse.w;
        for x in w.iter_mut().filter(|x| **x > 0.0) {
            *x = f64::from((*x as f32).max(f32::MIN_POSITIVE));
        }
        let phi = phi(&self.full, &self.spectrum(&w));
        (w, phi)
    }

    fn kept(w: &[f64]) -> impl Iterator<Item = usize> + '_ {
        (0..w.len()).filter(|&e| w[e] > 0.0)
    }

    fn spectrum(&self, w: &[f64]) -> Spectrum {
        let edges: Vec<Edge> = Self::kept(w)
            .map(|e| (self.edges[e].0, self.edges[e].1, w[e]))
            .collect();
        Spectrum::of(self.n, &edges)
    }

    fn connected(&self, w: &[f64]) -> bool {
        is_connected(
            self.n,
            Self::kept(w).map(|e| (self.edges[e].0, self.edges[e].1)),
        )
    }

    /// The first-order change in Φ of adding each edge left out at its full
    /// weight, lowest (best) first.
    fn addition_gains(&self, g: &Gradient, w: &[f64]) -> Vec<(f64, usize)> {
        let mut gains: Vec<(f64, usize)> = (0..w.len())
            .filter(|&e| w[e] == 0.0)
            .map(|e| (g.at(&self.edges[e]) * self.edges[e].2, e))
            .collect();
        gains.sort_by(|a, b| a.0.total_cmp(&b.0));
        gains
    }

    /// The maximum spanning tree, at the full weights.
    fn spanning_tree(&self) -> Vec<f64> {
        let mut w = vec![0.0; self.edges.len()];
        for e in maximum_spanning_tree(self.n, self.edges) {
            w[e] = self.edges[e].2;
        }
        w
    }

    /// Add edges at their full weight, in `add_rounds` batches, those
    /// whose weight lowers Φ most first, until `keep` are kept.
    fn add_to(&self, w: &mut [f64], keep: usize) {
        let mut kept = Self::kept(w).count();
        let mut rounds = self.args.add_rounds.max(1);
        while kept < keep {
            let g = Gradient::new(&self.full, &self.spectrum(w));
            let batch = (keep - kept).div_ceil(rounds);
            for (_, e) in self.addition_gains(&g, w).into_iter().take(batch) {
                w[e] = self.edges[e].2;
            }
            kept += batch;
            rounds = (rounds - 1).max(1);
        }
    }

    /// `w` with every weight scaled by `(n − 1) / Σ w̃ R` where that lowers
    /// Φ (a scaled spectrum is the same eigenvectors, the eigenvalues
    /// scaled: no decomposition more).
    fn rescaled(&self, mut w: Vec<f64>) -> Sparse {
        let spectrum = self.spectrum(&w);
        let y = self.full.weighted_vectors(|l| l.sqrt().recip());
        let total: f64 = Self::kept(&w)
            .map(|e| w[e] * (y.row(self.edges[e].0) - y.row(self.edges[e].1)).norm_squared())
            .sum();
        let scale = (self.n - 1) as f64 / total;
        if scale.is_finite() && scale > 0.0 {
            let scaled = spectrum.scaled(scale);
            if phi(&self.full, &scaled) < phi(&self.full, &spectrum) {
                w.iter_mut().for_each(|x| *x *= scale);
                return Sparse {
                    w,
                    spectrum: scaled,
                };
            }
        }
        Sparse { w, spectrum }
    }

    /// One swap of a kept edge for one left out (taking the removed one's
    /// scale, `w̃/w`), the best of a few checked exactly that lowers Φ and
    /// keeps the graph connected; `None` when none does.
    fn swap(&self, w: &[f64], g: &Gradient) -> Option<Sparse> {
        // First order: removing e changes Φ by −∂Φ/∂w̃ₑ · w̃ₑ; adding k at
        // weight x by ∂Φ/∂w̃ₖ · x.
        let mut removals: Vec<(f64, usize)> = Self::kept(w)
            .map(|e| (-g.at(&self.edges[e]) * w[e], e))
            .collect();
        removals.sort_by(|a, b| a.0.total_cmp(&b.0));
        removals.truncate(SWAP_CANDIDATES);
        let mut additions = self.addition_gains(g, w);
        additions.truncate(SWAP_CANDIDATES);
        let scale = |e: usize| w[e] / self.edges[e].2;
        let mut pairs: Vec<(f64, usize, usize)> = removals
            .iter()
            .flat_map(|&(dr, e)| {
                additions
                    .iter()
                    .map(move |&(da, k)| (dr + da * scale(e), e, k))
            })
            .collect();
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut best: Option<(f64, Sparse)> = None;
        let mut checked = 0;
        for (_, e, k) in pairs {
            if checked == self.args.exact_checks {
                break;
            }
            let mut trial = w.to_vec();
            trial[k] = self.edges[k].2 * scale(e);
            trial[e] = 0.0;
            if !self.connected(&trial) {
                continue;
            }
            checked += 1;
            let spectrum = self.spectrum(&trial);
            let p = phi(&self.full, &spectrum);
            if p < g.phi && best.as_ref().is_none_or(|(b, _)| p < *b) {
                best = Some((p, Sparse { w: trial, spectrum }));
            }
        }
        best.map(|(_, sparse)| sparse)
    }

    /// Fit the kept weights from `start` (whose gradient is `g`): Adam on
    /// their logs, `fit_steps` steps; the weights with the lowest Φ seen
    /// are kept, with their spectrum.
    fn fit(&self, start: Sparse, mut g: Gradient) -> Sparse {
        let ceiling = (MAX_WEIGHT_GAIN * self.edges.iter().map(|e| e.2).fold(0.0, f64::max)).ln();
        let kept: Vec<usize> = Self::kept(&start.w).collect();
        let mut w = start.w.clone();
        let mut theta: Vec<f64> = kept.iter().map(|&e| w[e].ln()).collect();
        let (mut m1, mut m2) = (vec![0.0; kept.len()], vec![0.0; kept.len()]);
        let (b1, b2) = ADAM_BETAS;
        let mut best = (g.phi, start);
        for step in 1..=self.args.fit_steps {
            for (q, &e) in kept.iter().enumerate() {
                let grad = g.at(&self.edges[e]) * w[e];
                m1[q] = b1 * m1[q] + (1.0 - b1) * grad;
                m2[q] = b2 * m2[q] + (1.0 - b2) * grad * grad;
                let m1_hat = m1[q] / (1.0 - b1.powi(step as i32));
                let m2_hat = m2[q] / (1.0 - b2.powi(step as i32));
                theta[q] -= self.args.fit_rate * m1_hat / (m2_hat.sqrt() + 1e-12);
                theta[q] = theta[q].min(ceiling);
                w[e] = theta[q].exp();
            }
            let spectrum = self.spectrum(&w);
            g = Gradient::new(&self.full, &spectrum);
            if g.phi < best.0 {
                best = (
                    g.phi,
                    Sparse {
                        w: w.clone(),
                        spectrum,
                    },
                );
            }
        }
        best.1
    }
}
