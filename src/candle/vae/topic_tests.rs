//! Tests for [`level_llik`]: an exact, deterministic score per level that
//! trains nothing and tracks what training optimizes.

use super::{level_llik, train_mixed, TrainConfig};
use crate::candle::decoder::gaussian_nb::GaussianNbDecoder;
use crate::candle::loss::topic_likelihood;
use crate::candle::traits::model::{DecoderModuleT, EncoderModuleT};
use crate::candle::vae::smooth_topics;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Init, VarBuilder, VarMap};
use nalgebra::DMatrix;
use std::sync::atomic::AtomicBool;

/// A deterministic encoder: a linear map of `ln(1 + x)`, with a noise
/// term in training mode only, so evaluation mode is the mean.
struct Lin(candle_nn::Linear, usize);

impl EncoderModuleT for Lin {
    fn forward_t(
        &self,
        x: &Tensor,
        _x0: Option<&Tensor>,
        train: bool,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        use candle_nn::Module;
        let z = self.0.forward(&(x + 1.0)?.log()?)?;
        let z = if train {
            (z.randn_like(0.0, 1.0)? + z)?
        } else {
            z
        };
        Ok((z, Tensor::zeros(x.dim(0)?, x.dtype(), x.device())?))
    }
    fn dim_latent(&self) -> usize {
        self.1
    }
}

fn setup(d: usize, k: usize) -> (VarMap, Lin, GaussianNbDecoder) {
    let dev = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let w = vb
        .pp("enc")
        .get_with_hints((k, d), "weight", Init::Const(0.1))
        .unwrap();
    let dec = GaussianNbDecoder::new(d, k, vb.pp("dec")).unwrap();
    (varmap, Lin(candle_nn::Linear::new(w, None), k), dec)
}

fn x(n: usize, d: usize) -> DMatrix<f32> {
    DMatrix::from_fn(n, d, |i, j| (1 + (i * 3 + j * 5) % 7) as f32)
}

/// The mean over every row, whatever the minibatch, and the same as one
/// pass over all rows by hand; deterministic, though training mode is not.
#[test]
fn level_llik_is_the_exact_mean_over_rows() {
    let (n, d, k) = (10, 6, 2);
    let (_vm, enc, dec) = setup(d, k);
    let x = x(n, d);
    let dev = Device::Cpu;
    let levels = [(&x, None, &x)];
    let by3 = level_llik(&levels, &enc, std::slice::from_ref(&dec), &dev, 0.1, 3).unwrap();
    let by64 = level_llik(&levels, &enc, std::slice::from_ref(&dec), &dev, 0.1, 64).unwrap();
    let again = level_llik(&levels, &enc, std::slice::from_ref(&dec), &dev, 0.1, 3).unwrap();
    assert_eq!(by3, again);
    assert!(
        (by3[0] - by64[0]).abs() < 1e-4 * by3[0].abs(),
        "{by3:?} {by64:?}"
    );

    let flat: Vec<f32> = (0..n)
        .flat_map(|i| x.row(i).iter().copied().collect::<Vec<_>>())
        .collect();
    let t = Tensor::from_vec(flat, (n, d), &dev).unwrap();
    let (z, _) = enc.forward_t(&t, None, false).unwrap();
    let z = smooth_topics(z, 0.1).unwrap();
    let (_, llik) = dec.forward_with_llik(&z, &t, &topic_likelihood).unwrap();
    let want = llik.sum_all().unwrap().to_scalar::<f32>().unwrap() / n as f32;
    assert!(
        (by3[0] - want).abs() < 1e-4 * want.abs(),
        "{by3:?} vs {want}"
    );
}

/// One entry per level; scoring moves no variable.
#[test]
fn level_llik_scores_each_level_and_trains_nothing() {
    let (d, k) = (6, 2);
    let (vm, enc, dec) = setup(d, k);
    let dec2 = {
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        GaussianNbDecoder::new(d, k, vb.pp("dec2")).unwrap()
    };
    let (a, b) = (x(10, d), x(4, d));
    let before: Vec<Vec<f32>> = vm
        .all_vars()
        .iter()
        .map(|v| v.flatten_all().unwrap().to_vec1().unwrap())
        .collect();
    let ll = level_llik(
        &[(&a, None, &a), (&b, None, &b)],
        &enc,
        &[dec, dec2],
        &Device::Cpu,
        0.0,
        4,
    )
    .unwrap();
    assert_eq!(ll.len(), 2);
    assert!(ll.iter().all(|v| v.is_finite()));
    let after: Vec<Vec<f32>> = vm
        .all_vars()
        .iter()
        .map(|v| v.flatten_all().unwrap().to_vec1().unwrap())
        .collect();
    assert_eq!(before, after);
}

/// Training raises it: the score tracks what training optimizes.
#[test]
fn level_llik_rises_with_training() {
    let (d, k) = (6, 2);
    let (vm, mut enc, dec) = setup(d, k);
    let x = x(12, d);
    let dev = Device::Cpu;
    let levels = [(&x, None, &x)];
    let decs = std::slice::from_ref(&dec);
    let before = level_llik(&levels, &enc, decs, &dev, 0.0, 8).unwrap()[0];
    let stop = AtomicBool::new(false);
    let config = TrainConfig {
        parameters: &vm,
        dev: &dev,
        epochs: 100,
        gpu_mem_fraction: None,
        minibatch_size: 4,
        learning_rate: 0.02,
        topic_smoothing: 0.0,
        grad_clip: 0.0,
        stop: &stop,
        seed: 1,
        loss_hook: None,
    };
    train_mixed(&levels, &mut enc, decs, &config).unwrap();
    let after = level_llik(&levels, &enc, decs, &dev, 0.0, 8).unwrap()[0];
    assert!(after > before, "{before} -> {after}");
}
