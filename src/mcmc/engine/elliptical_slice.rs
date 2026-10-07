use rand::rngs::SmallRng;
use rand::{Rng, RngExt, SeedableRng};
use rayon::prelude::*;
use std::f32::consts::PI;

use super::chain::McmcChain;
use super::runner::keep_flags;
use super::traits::EssParam;

/// Hard cap on bracket-shrinkage iterations inside [`elliptical_slice_step`].
/// In practice ~5–10 evals suffice; under stochastic likelihoods (e.g. rayon
/// parallel-summed f32) the shrinkage can stall near φ=0 if proposal lnpdf
/// drifts within the slice threshold. When this cap is hit we fall back to
/// the current state — well-defined because `cur_lnpdf > hh` is guaranteed
/// by `hh = ln(U) + cur_lnpdf` with `U ∈ (0,1)`.
pub(crate) const MAX_BRACKET_ITERS: usize = 64;

/// Bracket-width floor; if `phi_max − phi_min` falls below this the proposal
/// is numerically indistinguishable from current — same fallback as the
/// iteration cap.
pub(crate) const BRACKET_MIN_WIDTH: f32 = 1e-6;

/// One ESS transition. Returns `(new_params, new_lnpdf)`.
///
/// - `current`: current parameter value f
/// - `prior_sample`: a draw ν from the prior (caller handles Cholesky, etc.)
/// - `lnpdf`: log-likelihood function (just likelihood, not prior)
/// - `cur_lnpdf`: cached log-likelihood at `current`
/// - `rng`: random number generator
pub fn elliptical_slice_step<P: EssParam>(
    current: &P,
    prior_sample: &P,
    lnpdf: &impl Fn(&P) -> f32,
    cur_lnpdf: f32,
    rng: &mut impl Rng,
) -> (P, f32) {
    // 1. Choose ellipse and log-likelihood threshold
    let u: f32 = rng.random();
    let hh = u.ln() + cur_lnpdf;

    // 2. Draw initial proposal angle and define bracket
    let phi: f32 = rng.random_range(0.0..2.0 * PI);
    let mut phi_min = phi - 2.0 * PI;
    let mut phi_max = phi;

    // 3. Slice sampling loop with shrinkage + safety cap
    let mut angle = phi;
    for _ in 0..MAX_BRACKET_ITERS {
        let proposal = current.linear_combine(angle.cos(), prior_sample, angle.sin());
        let new_lnpdf = lnpdf(&proposal);

        if new_lnpdf > hh {
            return (proposal, new_lnpdf);
        }

        if angle < 0.0 {
            phi_min = angle;
        } else {
            phi_max = angle;
        }
        if phi_max - phi_min < BRACKET_MIN_WIDTH {
            break;
        }
        angle = rng.random_range(phi_min..phi_max);
    }

    // Fallback: bracket exhausted or collapsed. The current state is in
    // the slice by construction, so accept it.
    (current.clone(), cur_lnpdf)
}

/// ESS chain runner configuration.
#[derive(Clone, Debug)]
pub struct EssSampler {
    pub n_samples: usize,
    pub warmup: usize,
    /// Thinning interval; `0` is read as `1`.
    pub thin: usize,
    pub seed: u64,
}

impl EssSampler {
    pub fn new(n_samples: usize, warmup: usize) -> Self {
        Self {
            n_samples,
            warmup,
            thin: 1,
            seed: 42,
        }
    }

    /// Run a single ESS chain.
    ///
    /// - `lnpdf`: log-likelihood function
    /// - `prior_draw`: generates a sample from the prior N(0, Σ)
    /// - `init`: initial parameter value
    pub fn run<P: EssParam>(
        &self,
        lnpdf: &impl Fn(&P) -> f32,
        prior_draw: &impl Fn(&mut SmallRng) -> P,
        init: &P,
    ) -> McmcChain<P> {
        let mut rng = SmallRng::seed_from_u64(self.seed);

        let mut current = init.clone();
        let mut cur_lnpdf = lnpdf(&current);

        let mut samples = Vec::with_capacity(self.n_samples);
        let mut log_likelihoods = Vec::with_capacity(self.n_samples);

        for keep in keep_flags(self.warmup, self.n_samples, self.thin) {
            let nu = prior_draw(&mut rng);
            let (new, new_ll) = elliptical_slice_step(&current, &nu, lnpdf, cur_lnpdf, &mut rng);
            current = new;
            cur_lnpdf = new_ll;

            if keep {
                samples.push(current.clone());
                log_likelihoods.push(cur_lnpdf);
            }
        }

        McmcChain {
            samples,
            log_likelihoods,
        }
    }

    /// Run multiple independent chains in parallel via rayon.
    /// Each chain gets `seed + chain_idx` for reproducibility.
    pub fn run_parallel<P: EssParam + Send + Sync>(
        &self,
        n_chains: usize,
        lnpdf: &(impl Fn(&P) -> f32 + Sync),
        prior_draw: &(impl Fn(&mut SmallRng) -> P + Sync),
        init: &P,
    ) -> Vec<McmcChain<P>> {
        (0..n_chains)
            .into_par_iter()
            .map(|i| {
                let sampler = EssSampler {
                    seed: self.seed.wrapping_add(i as u64),
                    ..self.clone()
                };
                sampler.run(lnpdf, prior_draw, init)
            })
            .collect()
    }
}

#[cfg(test)]
#[path = "elliptical_slice_tests.rs"]
mod elliptical_slice_tests;
