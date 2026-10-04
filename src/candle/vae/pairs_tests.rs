//! Tests for the peer-pair penalty: the latent distances, the hinge, the pair
//! batches a minibatch takes, the checks before training, and that it pushes
//! a close pair apart, from one gradient step to a whole training run.

use super::{
    check_pairs, latent_distance, pair_batch, pair_hinge, quantile_distance, LevelPairs, PairMetric,
};
use crate::candle::convert::to_host;
use candle_core::{Device, Tensor, Var};

fn t(rows: &[&[f32]]) -> Tensor {
    let n = rows.len();
    let d = rows[0].len();
    let flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    Tensor::from_vec(flat, (n, d), &Device::Cpu).unwrap()
}

fn v(x: &Tensor) -> Vec<f32> {
    to_host(x).unwrap()
}

/// Hellinger on θ, from the log θ a topic encoder emits: 0 for the same
/// composition, 1 for disjoint ones.
#[test]
fn hellinger_reads_log_theta() {
    let ln = |p: f32| p.max(1e-30).ln();
    let a = t(&[&[ln(1.0), ln(0.0)], &[ln(0.5), ln(0.5)]]);
    let b = t(&[&[ln(0.0), ln(1.0)], &[ln(0.5), ln(0.5)]]);
    let d = v(&latent_distance(&a, &b, PairMetric::Hellinger).unwrap());
    assert!((d[0] - 1.0).abs() < 1e-5, "{d:?}");
    assert!(d[1].abs() < 1e-5, "{d:?}");
}

#[test]
fn euclidean_reads_z() {
    let a = t(&[&[0.0, 0.0], &[1.0, 1.0]]);
    let b = t(&[&[3.0, 4.0], &[1.0, 1.0]]);
    let d = v(&latent_distance(&a, &b, PairMetric::Euclidean).unwrap());
    assert!((d[0] - 5.0).abs() < 1e-5 && d[1].abs() < 1e-5, "{d:?}");
}

/// The distance at quantile `q` among all pairs of rows: a latent's own scale,
/// from which a margin is set. Points 0, 1, 2, 3 on a line have pair distances
/// 1, 1, 1, 2, 2, 3.
#[test]
fn quantile_distance_over_all_pairs() {
    let z = t(&[&[0.0], &[1.0], &[2.0], &[3.0]]);
    let at = |q| quantile_distance(&z, PairMetric::Euclidean, q).unwrap();
    assert_eq!(at(0.0), 1.0);
    assert_eq!(at(0.5), 1.0);
    assert_eq!(at(0.6), 2.0);
    assert_eq!(at(1.0), 3.0);
    assert!(quantile_distance(&t(&[&[0.0]]), PairMetric::Euclidean, 0.5).is_err());
    assert!(quantile_distance(&z, PairMetric::Euclidean, 1.5).is_err());
}

/// Past a few hundred thousand pairs it samples them: reproducibly, and close
/// to the exact quantile. 800 evenly spaced points on [0, 1] have a quarter of
/// their pair distances below 1 − √0.75 ≈ 0.134.
#[test]
fn quantile_distance_samples_large_levels() {
    let n = 800;
    let rows: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
    let z = Tensor::from_vec(rows, (n, 1), &Device::Cpu).unwrap();
    let a = quantile_distance(&z, PairMetric::Euclidean, 0.25).unwrap();
    let b = quantile_distance(&z, PairMetric::Euclidean, 0.25).unwrap();
    assert_eq!(a, b, "the same sample every time");
    assert!((a - 0.134).abs() < 0.005, "{a}");
}

/// Weighted mean of max(0, 1 − d/margin)²: pairs already apart cost nothing,
/// and the shortfall counts as a fraction of the margin, so one λ pulls alike
/// in a Hellinger latent (margins under 1) and a Euclidean one (margins of
/// several units), and across levels.
#[test]
fn hinge_charges_only_pairs_inside_the_margin() {
    let hinge = |d: [f32; 3], margin: f32| {
        let d = Tensor::new(&d, &Device::Cpu).unwrap();
        let w = Tensor::new(&[1.0f32, 2.0, 1.0], &Device::Cpu).unwrap();
        pair_hinge(&d, &w, margin)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    };
    // (1·1² + 2·0.5² + 1·0) / (1 + 2 + 1)
    let h = hinge([0.0, 0.5, 2.0], 1.0);
    assert!((h - 1.5 / 4.0).abs() < 1e-6, "{h}");
    let scaled = hinge([0.0, 2.0, 8.0], 4.0);
    assert!((scaled - h).abs() < 1e-6, "scale-free: {scaled} vs {h}");
    assert_eq!(
        hinge([0.0, 0.5, 2.0], 0.0),
        0.0,
        "a zero margin asks nothing"
    );
}

/// Each minibatch takes the next `batch` pairs, wrapping around, so every
/// pair is seen as training proceeds.
#[test]
fn pair_batches_cycle_through_the_pairs() {
    assert_eq!(pair_batch(5, 2, 0), vec![0, 1]);
    assert_eq!(pair_batch(5, 2, 1), vec![2, 3]);
    assert_eq!(pair_batch(5, 2, 2), vec![4, 0]);
    assert_eq!(pair_batch(3, 10, 0), vec![0, 1, 2]);
    assert!(pair_batch(0, 4, 7).is_empty());
}

/// Pairs are checked once, before training: rows in range, weights finite
/// and positive (a zero-weight batch would divide by zero, a negative weight
/// would pull the pair together), and a finite, non-negative margin.
#[test]
fn pairs_are_checked_before_training() {
    let level = |pairs: Vec<(u32, u32, f32)>, margin: f32| LevelPairs { pairs, margin };
    let ok = level(vec![(0, 1, 1.0), (2, 3, 0.5)], 0.3);
    assert!(check_pairs(&ok, 4).is_ok());
    assert!(
        check_pairs(&ok, 3).is_err(),
        "row 3 is out of range for 3 rows"
    );
    for bad in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(
            check_pairs(&level(vec![(0, 1, bad)], 0.3), 4).is_err(),
            "weight {bad} should be refused"
        );
    }
    for bad in [-0.1, f32::NAN, f32::INFINITY] {
        assert!(
            check_pairs(&level(vec![(0, 1, 1.0)], bad), 4).is_err(),
            "margin {bad} should be refused"
        );
    }
    assert!(
        check_pairs(&LevelPairs::default(), 0).is_ok(),
        "a level without pairs"
    );
}

/// The penalty's gradient moves a close pair apart: one step against it
/// raises their distance.
#[test]
fn the_penalty_pushes_a_close_pair_apart() {
    let za = Var::from_tensor(&t(&[&[0.0, 0.0]])).unwrap();
    let zb = Var::from_tensor(&t(&[&[0.1, 0.0]])).unwrap();
    let w = Tensor::new(&[1.0f32], &Device::Cpu).unwrap();
    let d = latent_distance(za.as_tensor(), zb.as_tensor(), PairMetric::Euclidean).unwrap();
    let before = v(&d)[0];
    let grads = pair_hinge(&d, &w, 1.0).unwrap().backward().unwrap();
    let step = |x: &Var| {
        let g = grads.get(x).unwrap().affine(0.1, 0.0).unwrap();
        x.as_tensor().sub(&g).unwrap()
    };
    let after = v(&latent_distance(&step(&za), &step(&zb), PairMetric::Euclidean).unwrap())[0];
    assert!(after > before, "{before} -> {after}");
}

/// Labelled pairs are often at the same point: the penalty's gradient there
/// must stay finite, or one such pair turns a whole training step into NaN.
#[test]
fn the_gradient_is_finite_at_zero_distance() {
    for metric in [PairMetric::Euclidean, PairMetric::Hellinger] {
        let za = Var::from_tensor(&t(&[&[-0.7, -0.7]])).unwrap();
        let zb = Var::from_tensor(&t(&[&[-0.7, -0.7]])).unwrap();
        let w = Tensor::new(&[1.0f32], &Device::Cpu).unwrap();
        let d = latent_distance(za.as_tensor(), zb.as_tensor(), metric).unwrap();
        let grads = pair_hinge(&d, &w, 1.0).unwrap().backward().unwrap();
        for x in [&za, &zb] {
            let g = v(grads.get(x).unwrap());
            assert!(g.iter().all(|v| v.is_finite()), "{metric:?}: {g:?}");
        }
    }
}

/// A transparent encoder for the revise tests: a learned linear map of
/// `ln(1 + x)` and a BatchNorm, no KL, so only the hinge moves it. The
/// BatchNorm starts at mean 0 and variance 1, nearly the identity, and its
/// running statistics are Vars an evaluation-mode pass reads.
struct LinearEncoder {
    lin: candle_nn::Linear,
    bn: crate::candle::nn::batch_norm::BatchNorm,
    k: usize,
    /// In training mode, emit the latent with no gradient: what the encoder
    /// learns then comes from passes in evaluation mode only.
    blind_in_training: bool,
}

impl crate::candle::traits::model::EncoderModuleT for LinearEncoder {
    fn forward_t(
        &self,
        x_nd: &Tensor,
        _x0_nd: Option<&Tensor>,
        train: bool,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        use candle_nn::{Module, ModuleT};
        let z = self.lin.forward(&(x_nd + 1.0)?.log()?)?;
        let z = self.bn.forward_t(&z, train)?;
        let z = if train && self.blind_in_training {
            z.detach()
        } else {
            z
        };
        let kl = Tensor::zeros(x_nd.dim(0)?, x_nd.dtype(), x_nd.device())?;
        Ok((z, kl))
    }

    fn dim_latent(&self) -> usize {
        self.k
    }
}

/// Two programs over `d` features; row 1 is row 0 with one feature shifted, a
/// small change of composition, so rows 0 and 1 start close in the latent.
fn two_programs(n: usize, d: usize) -> nalgebra::DMatrix<f32> {
    nalgebra::DMatrix::<f32>::from_fn(n, d, |i, j| {
        let r = if i == 1 { 0 } else { i };
        let program = if r % 2 == 0 { j < d / 2 } else { j >= d / 2 };
        let base = if program { 20.0 } else { 2.0 };
        let shift = if i == 1 && j == 0 { 4.0 } else { 0.0 };
        base + (r * 7 + j * 3) as f32 % 3.0 + shift
    })
}

/// An encoder under `enc` and a decoder under `dec` in one VarMap, the way a
/// trained model holds them, with a fixed starting point: candle cannot seed
/// its CPU generator.
struct Model {
    varmap: candle_nn::VarMap,
    enc: LinearEncoder,
    _dec: crate::candle::decoder::gaussian_nb::GaussianNbDecoder,
}

fn model(d: usize, k: usize, blind_in_training: bool) -> Model {
    use candle_core::DType;
    use candle_nn::{Init, VarBuilder, VarMap};
    let dev = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let w = vb
        .pp("enc")
        .get_with_hints((k, d), "weight", Init::Const(0.1))
        .unwrap();
    let b = vb
        .pp("enc")
        .get_with_hints(k, "bias", Init::Const(0.0))
        .unwrap();
    let bn =
        crate::candle::nn::batch_norm::batch_norm(k, Default::default(), &varmap, vb.pp("enc.bn"))
            .unwrap();
    let enc = LinearEncoder {
        lin: candle_nn::Linear::new(w, Some(b)),
        bn,
        k,
        blind_in_training,
    };
    let dec =
        crate::candle::decoder::gaussian_nb::GaussianNbDecoder::new(d, k, vb.pp("dec")).unwrap();
    Model {
        varmap,
        enc,
        _dec: dec,
    }
}

/// Every variable's values, by name, to compare before and after a revise.
fn snapshot(varmap: &candle_nn::VarMap, prefix: &str) -> Vec<(String, Vec<u32>)> {
    let data = varmap.data().lock().unwrap();
    let mut out: Vec<(String, Vec<u32>)> = data
        .iter()
        .filter(|(name, _)| name.starts_with(prefix))
        .map(|(name, var)| {
            let bits = v(&var.flatten_all().unwrap())
                .into_iter()
                .map(f32::to_bits)
                .collect();
            (name.clone(), bits)
        })
        .collect();
    out.sort();
    out
}

fn config<'a>(dev: &'a Device, stop: &'a AtomicBool, max_epochs: usize) -> ReviseConfig<'a> {
    ReviseConfig {
        dev,
        metric: PairMetric::Euclidean,
        topic_smoothing: 0.0,
        learning_rate: 0.05,
        max_epochs,
        batch: 4,
        grad_clip: 0.0,
        stop,
        verbose: false,
    }
}

fn distance(enc: &LinearEncoder, x: &nalgebra::DMatrix<f32>, a: usize, b: usize) -> f32 {
    use crate::candle::traits::model::EncoderModuleT;
    let row = |i: usize| {
        let r: Vec<f32> = x.row(i).iter().copied().collect();
        Tensor::from_vec(r, (1, x.ncols()), &Device::Cpu).unwrap()
    };
    let (za, _) = enc.forward_t(&row(a), None, false).unwrap();
    let (zb, _) = enc.forward_t(&row(b), None, false).unwrap();
    v(&latent_distance(&za, &zb, PairMetric::Euclidean).unwrap())[0]
}

use super::{encoder_trainable_vars, revise_encoder, ReviseConfig};
use std::sync::atomic::AtomicBool;

/// A labelled pair that starts close is pushed to its margin, and the run
/// stops there, before its budget: past the margin the hinge has no gradient.
#[test]
fn revise_pushes_a_labelled_pair_to_its_margin() {
    let x = two_programs(24, 8);
    let m = model(8, 2, false);
    let margin = 2.0;
    assert!(distance(&m.enc, &x, 0, 1) < margin);
    let pairs = [LevelPairs {
        pairs: vec![(0, 1, 1.0)],
        margin,
    }];
    let (dev, stop) = (Device::Cpu, AtomicBool::new(false));
    let trace = revise_encoder(
        &[(&x, None, &x)],
        &m.enc,
        &m.varmap,
        "enc",
        &pairs,
        &config(&dev, &stop, 5000),
    )
    .unwrap();
    assert!(distance(&m.enc, &x, 0, 1) >= margin * 0.999);
    let last = trace.hinge.len() - 1;
    assert_eq!(trace.hinge[last], vec![0.0]);
    assert_eq!(trace.satisfied[last], vec![1.0]);
    assert!(trace.hinge[0][0] > 0.0 && trace.satisfied[0] == vec![0.0]);
    assert!(last < 5000, "stopped once the hinge was zero");
}

/// The decoder and the BatchNorm running statistics are left out of the
/// optimizer: bit-identical after a revise that moved the encoder.
#[test]
fn revise_leaves_the_decoder_and_running_stats_bit_identical() {
    let x = two_programs(24, 8);
    let m = model(8, 2, false);
    let (dec0, enc0) = (snapshot(&m.varmap, "dec"), snapshot(&m.varmap, "enc"));
    let stats0 = snapshot(&m.varmap, "enc.bn.running");
    assert!(!dec0.is_empty());
    assert_eq!(stats0.len(), 2);
    let pairs = [LevelPairs {
        pairs: vec![(0, 1, 1.0), (2, 4, 0.5)],
        margin: 2.0,
    }];
    let (dev, stop) = (Device::Cpu, AtomicBool::new(false));
    revise_encoder(
        &[(&x, None, &x)],
        &m.enc,
        &m.varmap,
        "enc",
        &pairs,
        &config(&dev, &stop, 50),
    )
    .unwrap();
    assert_eq!(snapshot(&m.varmap, "dec"), dec0);
    assert_eq!(snapshot(&m.varmap, "enc.bn.running"), stats0);
    assert_ne!(snapshot(&m.varmap, "enc"), enc0, "the encoder moved");
}

/// Nothing to do changes nothing: no pairs, a level of pairs already past
/// their margin, or a zero margin. The trace holds the starting state only.
#[test]
fn revise_with_nothing_to_do_changes_nothing() {
    let x = two_programs(24, 8);
    let m = model(8, 2, false);
    let far = distance(&m.enc, &x, 0, 2);
    let cases = [
        vec![LevelPairs::default()],
        vec![],
        vec![LevelPairs {
            pairs: vec![(0, 2, 1.0)],
            margin: far * 0.5,
        }],
        vec![LevelPairs {
            pairs: vec![(0, 1, 1.0)],
            margin: 0.0,
        }],
    ];
    let before = snapshot(&m.varmap, "");
    let (dev, stop) = (Device::Cpu, AtomicBool::new(false));
    for pairs in cases {
        let trace = revise_encoder(
            &[(&x, None, &x)],
            &m.enc,
            &m.varmap,
            "enc",
            &pairs,
            &config(&dev, &stop, 100),
        )
        .unwrap();
        assert_eq!(trace.steps, 0, "{pairs:?}");
        assert_eq!(trace.hinge, vec![vec![0.0]], "{pairs:?}");
        assert_eq!(trace.satisfied, vec![vec![1.0]], "{pairs:?}");
        assert_eq!(snapshot(&m.varmap, ""), before, "{pairs:?}");
    }
}

/// A margin out of reach runs the whole budget and no more: one trace row per
/// epoch after the starting one, and as many steps as the epochs hold.
#[test]
fn revise_honours_the_epoch_budget() {
    let x = two_programs(24, 8);
    let m = model(8, 2, false);
    // 6 pairs in batches of 4: two steps an epoch.
    let pairs = [LevelPairs {
        pairs: (0..6).map(|i| (i, i + 6, 1.0)).collect(),
        margin: 1e6,
    }];
    let (dev, stop) = (Device::Cpu, AtomicBool::new(false));
    let trace = revise_encoder(
        &[(&x, None, &x)],
        &m.enc,
        &m.varmap,
        "enc",
        &pairs,
        &config(&dev, &stop, 3),
    )
    .unwrap();
    assert_eq!(trace.hinge.len(), 4);
    assert_eq!(trace.satisfied.len(), 4);
    assert_eq!(trace.steps, 6);
    assert!(trace.hinge.iter().all(|h| h[0] > 0.0));
}

/// Levels are revised together, each against its own margin; a level without
/// pairs reports a zero hinge and every pair satisfied.
#[test]
fn revise_traces_every_level() {
    let x = two_programs(24, 8);
    let coarse = two_programs(12, 8);
    let m = model(8, 2, false);
    let pairs = [
        LevelPairs {
            pairs: vec![(0, 1, 1.0)],
            margin: 2.0,
        },
        LevelPairs::default(),
    ];
    let (dev, stop) = (Device::Cpu, AtomicBool::new(false));
    let trace = revise_encoder(
        &[(&x, None, &x), (&coarse, None, &coarse)],
        &m.enc,
        &m.varmap,
        "enc",
        &pairs,
        &config(&dev, &stop, 5),
    )
    .unwrap();
    assert!(trace.hinge.iter().all(|h| h.len() == 2 && h[1] == 0.0));
    assert!(trace.satisfied.iter().all(|s| s[1] == 1.0));
}

/// The hinge measures pairs as the critique sees them, by the encoder's mean
/// in evaluation mode. Here training mode carries no gradient, so only an
/// evaluation-mode pass can move the encoder.
#[test]
fn revise_reads_the_evaluation_mode_latent() {
    let x = two_programs(24, 8);
    let m = model(8, 2, true);
    let before = snapshot(&m.varmap, "enc");
    let pairs = [LevelPairs {
        pairs: vec![(0, 1, 1.0)],
        margin: 10.0,
    }];
    let (dev, stop) = (Device::Cpu, AtomicBool::new(false));
    revise_encoder(
        &[(&x, None, &x)],
        &m.enc,
        &m.varmap,
        "enc",
        &pairs,
        &config(&dev, &stop, 2),
    )
    .unwrap();
    assert_ne!(snapshot(&m.varmap, "enc"), before);
}

/// Bad pairs, more pair levels than data levels, or a zero batch are refused
/// before any step.
#[test]
fn revise_checks_its_inputs() {
    let x = two_programs(24, 8);
    let m = model(8, 2, false);
    let (dev, stop) = (Device::Cpu, AtomicBool::new(false));
    let run = |pairs: &[LevelPairs], batch: usize| {
        let mut c = config(&dev, &stop, 5);
        c.batch = batch;
        revise_encoder(&[(&x, None, &x)], &m.enc, &m.varmap, "enc", pairs, &c)
    };
    let ok = LevelPairs {
        pairs: vec![(0, 1, 1.0)],
        margin: 1.0,
    };
    let out_of_range = LevelPairs {
        pairs: vec![(0, 24, 1.0)],
        margin: 1.0,
    };
    assert!(run(&[out_of_range], 4).is_err());
    assert!(run(&[ok.clone(), ok.clone()], 4).is_err());
    assert!(run(&[ok], 0).is_err());
}

/// The trainable set is the encoder's prefix less its running statistics; a
/// prefix that matches nothing is refused by a revise.
#[test]
fn encoder_trainable_vars_skip_running_stats() {
    let m = model(8, 2, false);
    let under_enc = snapshot(&m.varmap, "enc").len();
    // lin weight and bias, bn weight and bias
    assert_eq!(under_enc, 6);
    assert_eq!(encoder_trainable_vars(&m.varmap, "enc").len(), 4);
    let x = two_programs(24, 8);
    let pairs = [LevelPairs {
        pairs: vec![(0, 1, 1.0)],
        margin: 2.0,
    }];
    let (dev, stop) = (Device::Cpu, AtomicBool::new(false));
    let c = config(&dev, &stop, 5);
    assert!(revise_encoder(&[(&x, None, &x)], &m.enc, &m.varmap, "nope", &pairs, &c).is_err());
}
