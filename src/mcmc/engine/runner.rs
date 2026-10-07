use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;

use super::model::McmcModel;

/// Configuration for the MCMC runner.
#[derive(Clone, Debug)]
pub struct McmcConfig {
    /// Number of posterior samples to collect.
    pub n_samples: usize,
    /// Warmup (burn-in) iterations.
    pub warmup: usize,
    /// Thinning interval; `0` is read as `1`.
    pub thin: usize,
    /// RNG seed.
    pub seed: u64,
}

/// One flag per iteration, `true` where a draw is kept: `warmup` discarded, then
/// every `thin`-th draw until `n_samples` are kept (a `thin` of `0` is read as `1`).
/// Ends on the last kept draw, so no sweep runs only to be thrown away.
pub(super) fn keep_flags(
    warmup: usize,
    n_samples: usize,
    thin: usize,
) -> impl Iterator<Item = bool> {
    let thin = thin.max(1);
    let total = match n_samples {
        0 => Some(warmup),
        n => (n - 1)
            .checked_mul(thin)
            .and_then(|t| t.checked_add(warmup + 1)),
    }
    .expect("warmup + n_samples * thin overflows usize");
    (0..total).map(move |i| i >= warmup && (i - warmup).is_multiple_of(thin))
}

/// Run a single MCMC chain.
pub fn run_mcmc<M: McmcModel>(model: &M, config: &McmcConfig) -> M::Result {
    let mut rng = SmallRng::seed_from_u64(config.seed);
    let mut state = model.init(&mut rng);
    let mut samples = Vec::with_capacity(config.n_samples);

    for keep in keep_flags(config.warmup, config.n_samples, config.thin) {
        model.sweep(&mut state, &mut rng);
        if keep {
            samples.push(model.collect(&state));
        }
    }

    model.summarize(samples)
}

/// Run multiple independent chains in parallel.
/// Each chain gets `seed + chain_idx` for reproducibility.
pub fn run_mcmc_parallel<M: McmcModel + Sync>(
    model: &M,
    config: &McmcConfig,
    n_chains: usize,
) -> Vec<M::Result>
where
    M::State: Send,
    M::Result: Send,
{
    (0..n_chains)
        .into_par_iter()
        .map(|i| {
            let chain_config = McmcConfig {
                seed: config.seed.wrapping_add(i as u64),
                ..config.clone()
            };
            run_mcmc(model, &chain_config)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::keep_flags;

    #[test]
    fn keeps_every_thin_th_draw_after_warmup() {
        let kept: Vec<bool> = keep_flags(2, 3, 2).collect();
        let (f, t) = (false, true);
        assert_eq!(kept, [f, f, t, f, t, f, t]);
    }

    /// `thin: 0` would otherwise run only the warmup and keep nothing.
    #[test]
    fn a_zero_thin_is_read_as_one() {
        assert!(keep_flags(3, 4, 0).eq(keep_flags(3, 4, 1)));
        assert_eq!(keep_flags(3, 4, 0).filter(|&k| k).count(), 4);
    }

    #[test]
    #[should_panic(expected = "overflows usize")]
    fn an_overflowing_schedule_panics_rather_than_wrapping() {
        let _ = keep_flags(0, usize::MAX, 2);
    }
}
