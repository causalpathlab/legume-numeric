//! Peer-pair revision: pairs of samples another model holds apart are pushed
//! apart here. `senna critique` writes them, one set per cascade level, as row
//! indices into that level's training data.
//!
//! [`revise_encoder`] moves only the encoder, and only by the hinge
//! `Σ w·max(0, 1 − d(z_a, z_b)/m)² / Σ w`, summed over levels: zero once a pair
//! is at least the margin apart, so a revise ends when every pair is. No
//! likelihood enters the loss; the caller weighs it before and after, as a
//! gate on the revise. `d` is the distance the critique ranked in: Hellinger
//! on θ for an encoder that emits log θ, Euclidean on z for a Gaussian one.

use super::topic::LevelData;
use super::{clip_grads_and_step, smooth_topics};
use crate::candle::convert::to_1d;
use crate::candle::data::loader::{InMemoryArgs, InMemoryData};
use crate::candle::traits::model::EncoderModuleT;
use candle_core::{Device, Result, Tensor, Var};
use candle_nn::{AdamW, VarMap};
use log::{debug, info};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use std::sync::atomic::{AtomicBool, Ordering};

/// How a latent row is compared, matching the view the pairs came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PairMetric {
    /// Rows are log θ: `‖√θ_a − √θ_b‖ / √2`, in `[0, 1]`.
    Hellinger,
    /// Rows are z: `‖z_a − z_b‖`.
    Euclidean,
}

/// One level's pairs, `(row a, row b, weight)` in rows of the level's data,
/// and the distance at which they stop costing anything. Per level, because a
/// latent's spread differs by level: finer pseudobulks sit closer together.
/// The default, no pairs, is a level without any.
#[derive(Clone, Debug, Default)]
pub struct LevelPairs {
    pub pairs: Vec<(u32, u32, f32)>,
    pub margin: f32,
}

/// Before training: every row below `n_rows`, every weight finite and
/// positive, so a batch's weights never sum to zero and no pair is pulled
/// together, and a finite margin not below zero.
pub fn check_pairs(level: &LevelPairs, n_rows: usize) -> anyhow::Result<()> {
    let pairs = &level.pairs;
    if let Some(&(a, b, _)) = pairs
        .iter()
        .find(|&&(a, b, _)| a as usize >= n_rows || b as usize >= n_rows)
    {
        anyhow::bail!("pair ({a}, {b}) is out of range for a level of {n_rows} rows");
    }
    if let Some(&(_, _, w)) = pairs.iter().find(|&&(_, _, w)| !(w.is_finite() && w > 0.0)) {
        anyhow::bail!("pair weight {w} is not finite and positive");
    }
    anyhow::ensure!(
        level.margin.is_finite() && level.margin >= 0.0,
        "margin {} is not finite and non-negative",
        level.margin
    );
    Ok(())
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

/// Pairs [`quantile_distance`] measures at most; a larger level is sampled.
const QUANTILE_PAIRS: usize = 200_000;

/// The distance at quantile `q` among pairs of rows of `z`, in `metric`: a
/// latent's own scale, to set a level's margin from. Every pair when there are
/// at most [`QUANTILE_PAIRS`], else that many drawn with a fixed seed, so the
/// same latent always gives the same margin.
pub fn quantile_distance(z: &Tensor, metric: PairMetric, q: f32) -> anyhow::Result<f32> {
    let n = z.dim(0)?;
    anyhow::ensure!(
        n >= 2,
        "a quantile of pair distances needs two rows, not {n}"
    );
    anyhow::ensure!((0.0..=1.0).contains(&q), "quantile {q} is outside [0, 1]");
    let (a, b): (Vec<u32>, Vec<u32>) = if n * (n - 1) / 2 <= QUANTILE_PAIRS {
        (0..n as u32)
            .flat_map(|i| (i + 1..n as u32).map(move |j| (i, j)))
            .unzip()
    } else {
        let mut rng = SmallRng::seed_from_u64(0x5EED);
        (0..QUANTILE_PAIRS)
            .map(|_| {
                let i = rng.random_range(0..n);
                let j = (i + rng.random_range(1..n)) % n;
                (i as u32, j as u32)
            })
            .unzip()
    };
    let rows = |idx: Vec<u32>| -> Result<Tensor> {
        let len = idx.len();
        z.index_select(&Tensor::from_vec(idx, len, z.device())?, 0)
    };
    let mut d = latent_distance(&rows(a)?, &rows(b)?, metric)?.to_vec1::<f32>()?;
    let at = (q * (d.len() - 1) as f32).floor() as usize;
    let (_, &mut v, _) = d.select_nth_unstable_by(at, f32::total_cmp);
    Ok(v)
}

/// `Σ w·max(0, 1 − d/margin)² / Σ w`: the shortfall as a fraction of the
/// margin, so levels and metrics weigh alike in a sum of hinges. A zero margin
/// asks nothing.
pub fn pair_hinge(d: &Tensor, w: &Tensor, margin: f32) -> Result<Tensor> {
    if margin <= 0.0 {
        return d.zeros_like()?.sum_all();
    }
    let gap = d.affine(-1.0 / f64::from(margin), 1.0)?.relu()?;
    let num = (gap.sqr()? * w)?.sum_all()?;
    let den = w.sum_all()?;
    num / den
}

/// The pairs step `step` takes: the next `batch` of `n`, wrapping around.
pub fn pair_batch(n: usize, batch: usize, step: usize) -> Vec<usize> {
    if n == 0 || batch == 0 {
        return Vec::new();
    }
    let take = batch.min(n);
    let start = (step * take) % n;
    (0..take).map(|i| (start + i) % n).collect()
}

/// Settings for [`revise_encoder`].
pub struct ReviseConfig<'a> {
    pub dev: &'a Device,
    pub metric: PairMetric,
    /// As in training, so pairs are measured on the latent the decoder sees.
    pub topic_smoothing: f64,
    pub learning_rate: f32,
    /// Passes over every level's pairs at most; a revise stops sooner once
    /// every pair is at its margin.
    pub max_epochs: usize,
    /// Pairs per level per step. Each step encodes `2 · batch` rows a level.
    pub batch: usize,
    /// Global L2 gradient clip; 0 turns it off.
    pub grad_clip: f32,
    pub stop: &'a AtomicBool,
    /// Log each epoch at `info` rather than `debug`.
    pub verbose: bool,
}

/// What a revise did. Row 0 of `hinge` and `satisfied` is the state before
/// any step, row `e` the state after epoch `e`; each row has one entry per
/// level of data.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ReviseTrace {
    /// Each level's hinge over all its pairs; 0 for a level without any.
    pub hinge: Vec<Vec<f32>>,
    /// Each level's share of pairs at or beyond the margin; 1 for a level
    /// without any.
    pub satisfied: Vec<Vec<f32>>,
    /// Optimizer steps taken.
    pub steps: usize,
}

/// What [`revise_encoder`] optimizes: the variables of `varmap` under
/// `prefix` (`nn.enc` for the topic and Gaussian encoders), less any
/// BatchNorm running statistics. Those are Vars too, estimates of the data
/// rather than parameters: an evaluation-mode pass reads them detached
/// today, and leaving them out of the optimizer keeps a revise from moving
/// them should that change.
pub fn encoder_trainable_vars(varmap: &VarMap, prefix: &str) -> Vec<Var> {
    let is_running_stat = |name: &str| {
        ["running_mean", "running_var"]
            .iter()
            .any(|stat| name == *stat || name.ends_with(&format!(".{stat}")))
    };
    let data = varmap.data().lock().expect("VarMap lock poisoned");
    let mut vars: Vec<(&String, &Var)> = data
        .iter()
        .filter(|(name, _)| name.starts_with(prefix) && !is_running_stat(name))
        .collect();
    // A fixed order, so a revise does not depend on the map's.
    vars.sort_by(|a, b| a.0.cmp(b.0));
    vars.into_iter().map(|(_, var)| var.clone()).collect()
}

/// Move the encoder until every level's labelled pairs are at least their
/// margin apart, or the epoch budget runs out.
///
/// The loss is the sum over levels of [`pair_hinge`] on the next `batch`
/// pairs, measured on the evaluation-mode latent (the mean the critique
/// ranked, not a sample an encoder could spread by inflating its noise),
/// smoothed as in training. Only [`encoder_trainable_vars`]`(varmap,
/// encoder_prefix)` are optimized, so the decoder and the BatchNorm running
/// statistics stay as they are. `per_level[l]` holds the pairs of
/// `level_data[l]`; missing levels have none.
pub fn revise_encoder<Enc: EncoderModuleT>(
    level_data: &[LevelData],
    encoder: &Enc,
    varmap: &VarMap,
    encoder_prefix: &str,
    per_level: &[LevelPairs],
    config: &ReviseConfig,
) -> anyhow::Result<ReviseTrace> {
    anyhow::ensure!(config.batch > 0, "a revise needs a pair batch above zero");
    let vars = encoder_trainable_vars(varmap, encoder_prefix);
    anyhow::ensure!(
        !vars.is_empty(),
        "no trainable variables under prefix {encoder_prefix:?}"
    );
    anyhow::ensure!(
        per_level.len() <= level_data.len(),
        "pairs for {} levels, data for {}",
        per_level.len(),
        level_data.len()
    );
    for (level, (lp, &(input, _, _))) in per_level.iter().zip(level_data).enumerate() {
        check_pairs(lp, input.nrows()).map_err(|e| anyhow::anyhow!("level {level}: {e}"))?;
    }

    // Only levels with pairs are uploaded; the others are traced as done.
    let none = LevelPairs::default();
    let levels: Vec<(&LevelPairs, Option<InMemoryData>)> = level_data
        .iter()
        .enumerate()
        .map(|(level, &(input, null, _))| {
            let lp = per_level.get(level).unwrap_or(&none);
            let data = if lp.pairs.is_empty() || lp.margin <= 0.0 {
                None
            } else {
                let args = InMemoryArgs {
                    input,
                    input_null: null,
                    output: None,
                    output_null: None,
                };
                Some(InMemoryData::from_device(args, config.dev)?)
            };
            Ok((lp, data))
        })
        .collect::<anyhow::Result<_>>()?;

    let measure = |trace: &mut ReviseTrace| -> anyhow::Result<bool> {
        let mut hinge = Vec::with_capacity(levels.len());
        let mut satisfied = Vec::with_capacity(levels.len());
        for (lp, data) in &levels {
            let (h, s) = match data {
                Some(data) => level_hinge(encoder, data, lp, config)?,
                None => (0.0, 1.0),
            };
            hinge.push(h);
            satisfied.push(s);
        }
        let done = hinge.iter().all(|&h| h == 0.0);
        trace.hinge.push(hinge);
        trace.satisfied.push(satisfied);
        Ok(done)
    };

    let mut trace = ReviseTrace::default();
    if measure(&mut trace)? {
        return Ok(trace);
    }

    let steps_per_epoch = levels
        .iter()
        .filter(|(_, data)| data.is_some())
        .map(|(lp, _)| lp.pairs.len().div_ceil(config.batch))
        .max()
        .unwrap_or(0);
    let mut adam = AdamW::new_lr(vars, f64::from(config.learning_rate))?;

    for epoch in 0..config.max_epochs {
        for _ in 0..steps_per_epoch {
            let mut loss: Option<Tensor> = None;
            for (lp, data) in &levels {
                let Some(data) = data else { continue };
                let take = pair_batch(lp.pairs.len(), config.batch, trace.steps);
                let (d, w) = batch_distance(encoder, data, lp, &take, config)?;
                let h = pair_hinge(&d, &w, lp.margin)?;
                loss = Some(match loss {
                    Some(l) => (l + h)?,
                    None => h,
                });
            }
            if let Some(loss) = loss {
                clip_grads_and_step(&mut adam, &loss, f64::from(config.grad_clip))?;
            }
            trace.steps += 1;
            if config.stop.load(Ordering::Relaxed) {
                break;
            }
        }

        let done = measure(&mut trace)?;
        let msg = format!(
            "[revise epoch {epoch}] hinge={:?} satisfied={:?}",
            trace.hinge.last().unwrap(),
            trace.satisfied.last().unwrap()
        );
        if config.verbose {
            info!("{msg}");
        } else {
            debug!("{msg}");
        }
        if done || config.stop.load(Ordering::SeqCst) {
            break;
        }
    }
    Ok(trace)
}

/// Distances and weights of pairs `take` of `lp`, both ends in one
/// evaluation-mode forward pass.
fn batch_distance<Enc: EncoderModuleT>(
    encoder: &Enc,
    data: &InMemoryData,
    lp: &LevelPairs,
    take: &[usize],
    config: &ReviseConfig,
) -> anyhow::Result<(Tensor, Tensor)> {
    let n = take.len();
    let both: Vec<u32> = take
        .iter()
        .map(|&i| lp.pairs[i].0)
        .chain(take.iter().map(|&i| lp.pairs[i].1))
        .collect();
    let w: Vec<f32> = take.iter().map(|&i| lp.pairs[i].2).collect();
    let (x, null) = data.device_rows(&both)?;
    let (z, _) = encoder.forward_t(&x, null.as_ref(), false)?;
    let z = smooth_topics(z, config.topic_smoothing)?;
    let d = latent_distance(&z.narrow(0, 0, n)?, &z.narrow(0, n, n)?, config.metric)?;
    let w = to_1d(&w, d.device())?;
    Ok((d, w))
}

/// A level's hinge over all its pairs, and the share at or past the margin.
fn level_hinge<Enc: EncoderModuleT>(
    encoder: &Enc,
    data: &InMemoryData,
    lp: &LevelPairs,
    config: &ReviseConfig,
) -> anyhow::Result<(f32, f32)> {
    let (mut num, mut den, mut met) = (0f64, 0f64, 0usize);
    let idx: Vec<usize> = (0..lp.pairs.len()).collect();
    for take in idx.chunks(config.batch) {
        let (d, _) = batch_distance(encoder, data, lp, take, config)?;
        let d: Vec<f32> = d.detach().to_vec1()?;
        for (&i, &d) in take.iter().zip(&d) {
            let w = f64::from(lp.pairs[i].2);
            let gap = (1.0 - f64::from(d) / f64::from(lp.margin)).max(0.0);
            num += w * gap * gap;
            den += w;
            met += usize::from(d >= lp.margin);
        }
    }
    Ok(((num / den) as f32, met as f32 / lp.pairs.len() as f32))
}

#[cfg(test)]
#[path = "pairs_tests.rs"]
mod pairs_tests;
