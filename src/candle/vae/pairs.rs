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
pub enum PairMetric {
    /// Rows are log θ: `‖√θ_a − √θ_b‖ / √2`, in `[0, 1]`.
    Hellinger,
    /// Rows are z: `‖z_a − z_b‖`.
    Euclidean,
}

/// One level's pairs: `(row a, row b, weight)`, rows of the level's data. An
/// empty list is a level without pairs.
pub type LevelPairs = Vec<(u32, u32, f32)>;

/// Before training: every row below `n_rows`, and every weight finite and
/// positive, so a batch's weights never sum to zero and no pair is pulled
/// together.
pub fn check_pairs(pairs: &[(u32, u32, f32)], n_rows: usize) -> anyhow::Result<()> {
    if let Some(&(a, b, _)) = pairs
        .iter()
        .find(|&&(a, b, _)| a as usize >= n_rows || b as usize >= n_rows)
    {
        anyhow::bail!("pair ({a}, {b}) is out of range for a level of {n_rows} rows");
    }
    if let Some(&(_, _, w)) = pairs.iter().find(|&&(_, _, w)| !(w.is_finite() && w > 0.0)) {
        anyhow::bail!("pair weight {w} is not finite and positive");
    }
    Ok(())
}

/// The penalty a trainer adds to its loss. `per_level[l]` holds level `l`'s
/// pairs. Its extra encoder pass over `2 · batch` rows per minibatch is not
/// counted by the CUDA minibatch-size probe (`gpu_mem_fraction`); keep `batch`
/// small next to the minibatch.
pub struct PairPenalty<'a> {
    pub per_level: &'a [LevelPairs],
    /// Weight of the penalty against the per-sample ELBO; 0 turns it off.
    pub lambda: f32,
    /// Distance at which a pair stops costing anything.
    pub margin: f32,
    pub metric: PairMetric,
    /// Pairs per minibatch.
    pub batch: usize,
}

/// Row-wise distance between two `[n, k]` latent blocks.
pub fn latent_distance(a: &Tensor, b: &Tensor, metric: PairMetric) -> Result<Tensor> {
    let diff = match metric {
        PairMetric::Hellinger => (a.exp()?.sqrt()? - b.exp()?.sqrt()?)?,
        PairMetric::Euclidean => (a - b)?,
    };
    // The floor keeps the gradient finite where a pair coincides, which is
    // exactly where labelled pairs start; it moves a distance by at most 1e-6.
    let d = (diff.sqr()?.sum(1)? + 1e-12)?.sqrt()?;
    match metric {
        PairMetric::Hellinger => d.affine(std::f64::consts::FRAC_1_SQRT_2, 0.0),
        PairMetric::Euclidean => Ok(d),
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
