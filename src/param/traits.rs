#[cfg(feature = "tensor")]
use crate::matrix::traits::CandleDataLoaderOps;
use crate::matrix::traits::{IoOps, MeltOps};

pub trait TwoStatInference: Inference + TwoStatParam {}

pub trait Inference {
    // Rows as tensors only when candle is linked (feature `tensor`).
    #[cfg(feature = "tensor")]
    type Mat: IoOps + MeltOps + CandleDataLoaderOps;
    #[cfg(not(feature = "tensor"))]
    type Mat: IoOps + MeltOps;
    type Scalar: Into<f32>;

    fn posterior_mean(&self) -> &Self::Mat;
    fn posterior_sd(&self) -> &Self::Mat;
    fn posterior_log_mean(&self) -> &Self::Mat;
    fn posterior_log_sd(&self) -> &Self::Mat;
    fn posterior_sample(&self) -> anyhow::Result<Self::Mat>;
    /// [`Self::posterior_sample`] drawn from `seed`: the same seed gives the
    /// same matrix, whatever the thread count, so a fit trained on it
    /// replays. Seeded per fixed-width chunk, like
    /// [`Self::posterior_log_sample`].
    ///
    /// The default refuses: an implementation outside this crate that
    /// predates the method has no seeded draw.
    fn posterior_sample_seeded(&self, seed: u64) -> anyhow::Result<Self::Mat> {
        let _ = seed;
        anyhow::bail!("this parameter type has no seeded posterior sample")
    }
    /// Draw a fresh sample of `log λ` per element. Delta-method
    /// approximation: `log λ ≈ Normal(posterior_log_mean,
    /// posterior_log_sd²)`. Caller must have called
    /// `calibrate_with(CalibrateTarget::All)` first so log_mean / log_sd
    /// are populated.
    ///
    /// `seed` makes the draw reproducible. It used to come from
    /// `rand::rng()` inside a `map_init`, which is OS-seeded AND partitioned
    /// by rayon, so a caller could not reproduce its own null distribution
    /// even with every other seed pinned. Seeding per CHUNK rather than per
    /// element keeps the draw parallel and independent of how rayon happens
    /// to split the work.
    fn posterior_log_sample(&self, seed: u64) -> anyhow::Result<Self::Mat>;

    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
}

/// Which posterior quantities to compute during calibration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrateTarget {
    /// Compute all: mean, sd, log_mean, log_sd
    All,
    /// Only posterior mean (a/b)
    MeanOnly,
    /// Posterior mean + log mean (digamma(a) - ln(b))
    MeanAndLogMean,
}

/// A parameter matrix with two types of statistics
/// with hyper parameters a0 and b0
pub trait TwoStatParam {
    type Mat;
    type Scalar;

    fn new(dims: (usize, usize), a0: Self::Scalar, b0: Self::Scalar) -> Self;
    fn add_stat(&mut self, add_a: &Self::Mat, add_b: &Self::Mat);
    fn update_stat(&mut self, update_a: &Self::Mat, update_b: &Self::Mat);
    fn update_stat_col(&mut self, update_a: &Self::Mat, update_b: &Self::Mat, k: usize);
    fn reset_stat(&mut self);

    /// Calibrate all posterior quantities (mean, sd, log_mean, log_sd)
    fn calibrate(&mut self) {
        self.calibrate_with(CalibrateTarget::All);
    }

    /// Calibrate only the posterior quantities specified by `target`.
    fn calibrate_with(&mut self, target: CalibrateTarget) {
        match target {
            CalibrateTarget::All => {
                self.map_calibrate_mean();
                self.map_calibrate_log_mean();
                self.map_calibrate_sd();
                self.map_calibrate_log_sd();
            }
            CalibrateTarget::MeanOnly => {
                self.map_calibrate_mean();
            }
            CalibrateTarget::MeanAndLogMean => {
                self.map_calibrate_mean();
                self.map_calibrate_log_mean();
            }
        }
    }

    fn map_calibrate_mean(&mut self);
    fn map_calibrate_sd(&mut self);
    fn map_calibrate_log_mean(&mut self);
    fn map_calibrate_log_sd(&mut self);
}

/// One Gamma draw per `(a, b)` pair, shape `a + ε` and rate `b + ε`, from
/// `seed`: one generator per fixed-width chunk, so the draw is a function of
/// the seed and the data shape, not of how rayon splits the work.
pub(crate) fn gamma_sample_seeded(a: &[f32], b: &[f32], seed: u64) -> anyhow::Result<Vec<f32>> {
    use rand::rngs::SmallRng;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Gamma};
    use rayon::prelude::*;
    const CHUNK: usize = 1024;
    let eps = 1e-8;
    let mut sampled = vec![0.0f32; a.len()];
    sampled
        .par_chunks_mut(CHUNK)
        .enumerate()
        .try_for_each(|(ci, out)| -> anyhow::Result<()> {
            let mut rng =
                SmallRng::seed_from_u64(seed ^ (ci as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let base = ci * CHUNK;
            for (k, o) in out.iter_mut().enumerate() {
                let shape = a[base + k] + eps;
                let scale = (b[base + k] + eps).recip();
                *o = Gamma::new(shape, scale)?.sample(&mut rng);
            }
            Ok(())
        })?;
    Ok(sampled)
}
