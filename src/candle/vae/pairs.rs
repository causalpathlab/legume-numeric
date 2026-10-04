//! Peer-pair penalty: pairs of samples another model holds apart are pushed
//! apart here. `senna critique` writes them, one set per cascade level, as row
//! indices into that level's training data.
//!
//! The penalty is `λ · Σ w·max(0, m − d(z_a, z_b))² / Σ w`: zero once a pair is
//! at least the margin apart, so it moves only pairs that are still too close,
//! and leaves the rest of the latent to the likelihood. `d` is the distance the
//! critique ranked in: Hellinger on θ for an encoder that emits log θ,
//! Euclidean on z for a Gaussian one.

use candle_core::{Result, Tensor};

/// How a latent row is compared, matching the view the pairs came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LatentMetric {
    /// Rows are log θ: `‖√θ_a − √θ_b‖ / √2`, in `[0, 1]`.
    Hellinger,
    /// Rows are z: `‖z_a − z_b‖`.
    Euclidean,
}

/// One level's pairs: row indices `a[i]`, `b[i]` into the level's data, and a
/// weight per pair.
#[derive(Clone, Debug, Default)]
pub struct LevelPairs {
    pub a: Vec<u32>,
    pub b: Vec<u32>,
    pub weight: Vec<f32>,
}

impl LevelPairs {
    pub fn len(&self) -> usize {
        self.a.len()
    }

    pub fn is_empty(&self) -> bool {
        self.a.is_empty()
    }
}

/// The penalty a trainer adds to its loss. `per_level[l]` is `None` for a level
/// without pairs.
pub struct PairPenalty<'a> {
    pub per_level: &'a [Option<LevelPairs>],
    /// Weight of the penalty against the per-sample ELBO.
    pub lambda: f32,
    /// Distance at which a pair stops costing anything.
    pub margin: f32,
    pub metric: LatentMetric,
    /// Pairs per minibatch.
    pub batch: usize,
}

/// Row-wise distance between two `[n, k]` latent blocks.
pub fn latent_distance(a: &Tensor, b: &Tensor, metric: LatentMetric) -> Result<Tensor> {
    let diff = match metric {
        LatentMetric::Hellinger => (a.exp()?.sqrt()? - b.exp()?.sqrt()?)?,
        LatentMetric::Euclidean => (a - b)?,
    };
    // The floor keeps the gradient finite where a pair coincides, which is
    // exactly where labelled pairs start; it moves a distance by at most 1e-6.
    let d = (diff.sqr()?.sum(1)? + 1e-12)?.sqrt()?;
    match metric {
        LatentMetric::Hellinger => d.affine(std::f64::consts::FRAC_1_SQRT_2, 0.0),
        LatentMetric::Euclidean => Ok(d),
    }
}

/// `Σ w·max(0, margin − d)² / Σ w`.
pub fn pair_hinge(d: &Tensor, w: &Tensor, margin: f32) -> Result<Tensor> {
    let gap = d.affine(-1.0, f64::from(margin))?.relu()?;
    let num = (gap.sqr()? * w)?.sum_all()?;
    let den = w.sum_all()?;
    num / den
}

/// The pairs minibatch `step` takes: the next `batch` of `n`, wrapping around.
pub fn pair_batch(n: usize, batch: usize, step: usize) -> Vec<usize> {
    if n == 0 || batch == 0 {
        return Vec::new();
    }
    let take = batch.min(n);
    let start = (step * take) % n;
    (0..take).map(|i| (start + i) % n).collect()
}

#[cfg(test)]
#[path = "pairs_tests.rs"]
mod pairs_tests;
