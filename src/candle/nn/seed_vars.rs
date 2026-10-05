//! Seeded initialisation of a `VarMap`'s linear weights.

use crate::matrix::rand_util::name_seed;
use candle_core::{Result, Tensor};
use candle_nn::VarMap;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// Re-draw every var in `varmap` — uniform in `±1/√fan_in` for weights,
/// zero for anything named `*.bias` — each from its own name-keyed
/// sub-stream of `seed`, so a run replays and adding a var never shifts
/// another's draw. Vars for which `skip(name)` holds keep their current
/// values (a normalisation layer's affine and statistics, say). candle's
/// `VarBuilder` initialises from an unseeded stream, which is why this
/// exists.
pub fn seed_uniform_vars(varmap: &VarMap, seed: u64, skip: impl Fn(&str) -> bool) -> Result<()> {
    let tbl = varmap.data().lock().unwrap();
    for (name, var) in tbl.iter() {
        if skip(name) {
            continue;
        }
        let dims = var.dims().to_vec();
        let n: usize = dims.iter().product();
        let draw: Vec<f32> = if name.ends_with(".bias") {
            vec![0f32; n]
        } else {
            let bound = (1.0 / *dims.last().unwrap_or(&1) as f64).sqrt();
            let mut rng = StdRng::seed_from_u64(name_seed(seed, name));
            (0..n)
                .map(|_| ((rng.random::<f64>() * 2.0 - 1.0) * bound) as f32)
                .collect()
        };
        var.set(&Tensor::from_vec(draw, dims.as_slice(), var.device())?)?;
    }
    Ok(())
}

/// Re-draw every var in `varmap` from its own name-keyed sub-stream of
/// `seed`, from the distribution it was declared with, so a model built
/// twice starts from the same weights. candle cannot seed its CPU
/// generator, and `VarBuilder` draws from it; this replaces those draws.
///
/// The declared initializations of this crate's models, by name and shape:
/// - a var whose values are all equal was declared constant (BatchNorm
///   affine and statistics, dispersions, backgrounds, a LoRA `v`): kept;
/// - `*.bias` beside a `*.weight`: uniform in `±1/√in`, `in` the weight's
///   input width (`candle_nn::linear`);
/// - `*modules.logits`: normal with the module branch's jitter;
/// - `*lora_u` `[n, r]`: normal with variance `1/r`;
/// - anything else: Kaiming-normal on its fan-in (`candle_nn::linear`'s
///   weights, the feature and topic tables, the attention query).
///
/// Vars for which `skip(name)` holds keep their current values.
pub fn seed_declared_vars(varmap: &VarMap, seed: u64, skip: impl Fn(&str) -> bool) -> Result<()> {
    use rand_distr::{Distribution, Normal};
    let tbl = varmap.data().lock().unwrap();
    for (name, var) in tbl.iter() {
        if skip(name) {
            continue;
        }
        let values: Vec<f32> = var.flatten_all()?.to_vec1()?;
        if values.windows(2).all(|w| w[0] == w[1]) {
            continue;
        }
        let dims = var.dims().to_vec();
        let n = values.len();
        let mut rng = StdRng::seed_from_u64(name_seed(seed, name));
        let normal = |std: f64, rng: &mut StdRng| -> Vec<f32> {
            let dist = Normal::new(0.0, std).expect("a finite positive deviation");
            (0..n).map(|_| dist.sample(rng) as f32).collect()
        };
        let sibling_in = name
            .strip_suffix(".bias")
            .and_then(|stem| tbl.get(&format!("{stem}.weight")))
            .and_then(|w| w.dims().get(1).copied());
        let draw: Vec<f32> = if let Some(in_dim) = sibling_in {
            let bound = 1.0 / (in_dim as f64).sqrt();
            (0..n)
                .map(|_| ((rng.random::<f64>() * 2.0 - 1.0) * bound) as f32)
                .collect()
        } else if name.ends_with("modules.logits") {
            normal(
                crate::candle::feature_embedding::INIT_LOGIT_JITTER,
                &mut rng,
            )
        } else if name.ends_with(crate::candle::lora::U_VAR_NAME) {
            normal((1.0 / *dims.last().unwrap_or(&1) as f64).sqrt(), &mut rng)
        } else {
            // candle's fan-in: the second dim times any receptive field.
            let fan_in = if dims.len() < 2 {
                1
            } else {
                dims[1] * dims.iter().skip(2).product::<usize>()
            };
            normal(2f64.sqrt() / (fan_in as f64).sqrt(), &mut rng)
        };
        var.set(&Tensor::from_vec(draw, dims.as_slice(), var.device())?)?;
    }
    Ok(())
}
