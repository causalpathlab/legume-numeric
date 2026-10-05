//! A run replays: two models built apart, seeded with one seed and trained
//! with another, end bit-identical, on the CPU, for the dense and the masked
//! family; and a different training seed gives a different model.

use crate::candle::decoder::gaussian_nb::GaussianNbDecoder;
use crate::candle::decoder::masked_etm::EmbeddedNbTopicDecoder;
use crate::candle::encoder::gaussian::{GaussianEncoder, GaussianEncoderArgs};
use crate::candle::encoder::indexed::{IndexedEmbeddingEncoder, IndexedEmbeddingEncoderArgs};
use crate::candle::nn::seed_declared_vars;
use crate::candle::vae::masked_topic::{train_masked, IndexedTrainConfig, MaskedTrainOpts};
use crate::candle::vae::topic::{train_mixed, TrainConfig};
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use nalgebra::DMatrix;
use std::sync::atomic::AtomicBool;

type Bits = Vec<(String, Vec<u32>)>;

fn bits(vm: &VarMap) -> Bits {
    let d = vm.data().lock().unwrap();
    let mut v: Bits = d
        .iter()
        .map(|(n, t)| {
            let b = t
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect();
            (n.clone(), b)
        })
        .collect();
    v.sort();
    v
}

fn counts(rows: usize, cols: usize) -> DMatrix<f32> {
    DMatrix::from_fn(rows, cols, |i, j| {
        ((i * 7 + j * 3) % 11) as f32 + 0.3 * (i + j) as f32
    })
}

/// A Gaussian encoder (batch norm, reparameterization noise) and an NB
/// decoder, built, seeded and trained for two epochs with `train_seed`.
fn dense_run(train_seed: u64) -> (Bits, Bits) {
    let (n, d, k) = (40, 8, 3);
    let x = counts(n, d);
    let dev = Device::Cpu;
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
    let mut enc = GaussianEncoder::new(
        GaussianEncoderArgs {
            n_features: d,
            n_latent: k,
            layers: &[16],
            feature_mean: None,
        },
        &vm,
        vb.pp("enc"),
    )
    .unwrap();
    let dec = GaussianNbDecoder::new(d, k, vb.pp("dec_0")).unwrap();
    seed_declared_vars(&vm, 7, |_| false).unwrap();
    let start = bits(&vm);
    let stop = AtomicBool::new(false);
    let config = TrainConfig {
        parameters: &vm,
        dev: &dev,
        epochs: 2,
        gpu_mem_fraction: None,
        minibatch_size: 16,
        learning_rate: 0.01,
        topic_smoothing: 0.0,
        // Clipped, as the topic and vae fits are: the clip reads a global
        // norm, which must not depend on the gradient store's order.
        grad_clip: 1.0,
        stop: &stop,
        seed: train_seed,
        loss_hook: None,
    };
    train_mixed(&[(&x, None, &x)], &mut enc, &[dec], &config).unwrap();
    (start, bits(&vm))
}

/// A masked encoder and decoder, built, seeded and trained for two epochs
/// with `train_seed`.
fn masked_run(train_seed: u64) -> (Bits, Bits) {
    let (d, p) = (8, 40);
    let x = counts(d, p);
    let mean: Vec<f32> = (0..d).map(|g| x.row(g).mean()).collect();
    let dev = Device::Cpu;
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
    let enc = IndexedEmbeddingEncoder::new(
        IndexedEmbeddingEncoderArgs {
            n_features: d,
            n_topics: 3,
            embedding_dim: 6,
            layers: &[16],
            attn_pool: true,
            n_gene_modules: 0,
            lora_rank: 0,
        },
        &vm,
        vb.pp("enc"),
    )
    .unwrap();
    let dec = EmbeddedNbTopicDecoder::new(3, enc.features_shared(), vb.pp("dec_0")).unwrap();
    seed_declared_vars(&vm, 7, |_| false).unwrap();
    let start = bits(&vm);
    let stop = AtomicBool::new(false);
    let config = IndexedTrainConfig {
        parameters: &vm,
        dev: &dev,
        epochs: 2,
        gpu_mem_fraction: None,
        minibatch_size: 16,
        learning_rate: 0.01,
        topic_smoothing: 0.0,
        stop: &stop,
        feature_mean: &mean,
        grad_clip: 0.0,
        feature_embedding_l2: 0.0,
        weight_decay: 0.0,
        feature_anchor: None,
    };
    let opts = MaskedTrainOpts {
        seed: train_seed,
        ..Default::default()
    };
    train_masked(&[(&x, None, &x)], &enc, &[dec], &config, 0.5, &opts).unwrap();
    (start, bits(&vm))
}

/// Fits A, B, then A again in one process: the second A matches the first,
/// so nothing an earlier fit leaves in the noise stream leaks into a later
/// one, and B differs.
#[test]
fn a_dense_run_replays_bit_for_bit() {
    let (start_a, end_a) = dense_run(1);
    let (_, end_b) = dense_run(2);
    let (start_a2, end_a2) = dense_run(1);
    assert_eq!(start_a, start_a2, "the seeded start");
    assert_eq!(end_a, end_a2, "two epochs from one seed, after another fit");
    assert_ne!(end_a, end_b, "another seed, another run");
}

#[test]
fn a_masked_run_replays_bit_for_bit() {
    let (start_a, end_a) = masked_run(1);
    let (start_b, end_b) = masked_run(1);
    assert_eq!(start_a, start_b, "the seeded start");
    assert_eq!(end_a, end_b, "two epochs from one seed");
    let (_, end_c) = masked_run(2);
    assert_ne!(end_a, end_c, "another seed, another run");
}

/// The redraw keeps a constant-declared var as it is and draws a weight at
/// its declared Kaiming-normal scale: standard deviation `√2 / √fan_in`.
#[test]
fn the_redraw_keeps_each_declared_initialization() {
    let dev = Device::Cpu;
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
    let _lin = candle_nn::linear(64, 200, vb.pp("lin")).unwrap();
    let _c = vb
        .get_with_hints(5, "phi", candle_nn::Init::Const(0.693))
        .unwrap();
    seed_declared_vars(&vm, 3, |_| false).unwrap();
    let data = vm.data().lock().unwrap();
    let phi: Vec<f32> = data["phi"].to_vec1().unwrap();
    assert!(phi.iter().all(|&v| v == 0.693f32));
    let w: Vec<f32> = data["lin.weight"].flatten_all().unwrap().to_vec1().unwrap();
    let sd = (w.iter().map(|x| x * x).sum::<f32>() / w.len() as f32).sqrt();
    let want = (2.0f32 / 64.0).sqrt();
    assert!((sd - want).abs() < 0.05 * want, "{sd} vs {want}");
    let b: Vec<f32> = data["lin.bias"].to_vec1().unwrap();
    assert!(b.iter().all(|x| x.abs() <= 1.0 / 8.0));
}

/// A one-element bias is a linear layer's, declared uniform: it is redrawn
/// from the seed, not kept as an unseeded draw.
#[test]
fn a_one_element_bias_is_redrawn_from_the_seed() {
    let build = || {
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        let _lin = candle_nn::linear(4, 1, vb.pp("out")).unwrap();
        seed_declared_vars(&vm, 5, |_| false).unwrap();
        bits(&vm)
    };
    assert_eq!(build(), build());
}

/// The seeded noise lives only while its guard does: the same seed replays,
/// and once the guard is gone the draws are candle's again.
#[test]
fn the_noise_stream_ends_with_its_guard() {
    use crate::candle::loss::{seed_noise, standard_normal_like};
    let t = candle_core::Tensor::zeros(64, DType::F32, &Device::Cpu).unwrap();
    let draw = || standard_normal_like(&t).unwrap().to_vec1::<f32>().unwrap();
    let a = {
        let _g = seed_noise(5);
        draw()
    };
    let b = {
        let _g = seed_noise(5);
        draw()
    };
    assert_eq!(a, b);
    assert_ne!(draw(), a, "after the guard, not the seeded stream");
}

/// Names match on path components: `gene_modules.logits` is a weight, drawn
/// at its Kaiming scale, not the module logits' jitter; an f64 model is
/// redrawn in its own dtype.
#[test]
fn the_redraw_reads_whole_names_and_any_dtype() {
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F64, &Device::Cpu);
    let _w = vb
        .pp("gene_modules")
        .get_with_hints((200, 64), "logits", candle_nn::init::DEFAULT_KAIMING_NORMAL)
        .unwrap();
    seed_declared_vars(&vm, 3, |_| false).unwrap();
    let data = vm.data().lock().unwrap();
    let w: Vec<f64> = data["gene_modules.logits"]
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let sd = (w.iter().map(|x| x * x).sum::<f64>() / w.len() as f64).sqrt();
    let want = (2.0f64 / 64.0).sqrt();
    assert!((sd - want).abs() < 0.05 * want, "{sd} vs {want}");
}
