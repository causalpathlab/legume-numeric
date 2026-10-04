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

/// A transparent encoder for the end-to-end test: a learned linear map of
/// `ln(1 + x)`, no batch norm and no KL, so training and evaluation agree and
/// only the penalty can pull a pair apart.
struct LinearEncoder {
    lin: candle_nn::Linear,
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
        use candle_nn::Module;
        let z = self.lin.forward(&(x_nd + 1.0)?.log()?)?;
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

/// End to end: two near-identical samples sit together in a trained latent;
/// labelled as a pair to push apart, training separates them.
#[test]
fn training_with_pairs_separates_a_labelled_pair() {
    use super::PairPenalty;
    use crate::candle::decoder::gaussian_nb::GaussianNbDecoder;
    use crate::candle::traits::model::EncoderModuleT;
    use crate::candle::vae::topic::{train_mixed_with_pairs, TrainConfig};
    use candle_core::DType;
    use candle_nn::{VarBuilder, VarMap};
    use nalgebra::DMatrix;
    use std::sync::atomic::AtomicBool;

    let (n, d, k) = (24usize, 8usize, 2usize);
    // Two programs. Row 1 is row 0 with one gene shifted: a small change of
    // composition.
    let x = DMatrix::<f32>::from_fn(n, d, |i, j| {
        let r = if i == 1 { 0 } else { i };
        let program = if r % 2 == 0 { j < d / 2 } else { j >= d / 2 };
        let base = if program { 20.0 } else { 2.0 };
        let shift = if i == 1 && j == 0 { 4.0 } else { 0.0 };
        base + (r * 7 + j * 3) as f32 % 3.0 + shift
    });
    let train = |with_pairs: bool| -> f32 {
        let dev = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        // A fixed starting point: candle cannot seed its CPU generator, and a
        // random one makes the run-to-run spread larger than the effect.
        let w0 = vb
            .pp("enc")
            .get_with_hints((k, d), "weight", candle_nn::Init::Const(0.1))
            .unwrap();
        let b0 = vb
            .pp("enc")
            .get_with_hints(k, "bias", candle_nn::Init::Const(0.0))
            .unwrap();
        let mut enc = LinearEncoder {
            lin: candle_nn::Linear::new(w0, Some(b0)),
            k,
            blind_in_training: false,
        };
        let dec = GaussianNbDecoder::new(d, k, vb.pp("dec")).unwrap();
        let stop = AtomicBool::new(false);
        let config = TrainConfig {
            parameters: &varmap,
            dev: &dev,
            epochs: 200,
            gpu_mem_fraction: None,
            minibatch_size: 8,
            learning_rate: 0.01,
            topic_smoothing: 0.0,
            grad_clip: 0.0,
            stop: &stop,
            loss_hook: None,
        };
        let pairs = [LevelPairs {
            pairs: vec![(0, 1, 1.0)],
            margin: 2.0,
        }];
        let penalty = PairPenalty {
            per_level: &pairs,
            lambda: 200.0,
            metric: PairMetric::Euclidean,
            batch: 1,
        };
        train_mixed_with_pairs(
            &[(&x, None, &x)],
            &mut enc,
            &[dec],
            &config,
            with_pairs.then_some(&penalty),
        )
        .unwrap();
        let rows = |i: usize| {
            let r: Vec<f32> = x.row(i).iter().copied().collect();
            Tensor::from_vec(r, (1, d), &dev).unwrap()
        };
        let (za, _) = enc.forward_t(&rows(0), None, false).unwrap();
        let (zb, _) = enc.forward_t(&rows(1), None, false).unwrap();
        v(&latent_distance(&za, &zb, PairMetric::Euclidean).unwrap())[0]
    };
    let without = train(false);
    let with = train(true);
    assert!(
        with > 0.3 && with > 3.0 * without,
        "with pairs {with}, without {without}"
    );
}

/// The penalty measures pairs as the critique sees them, by the encoder's
/// mean in evaluation mode. A sampled training-mode latent would let an
/// encoder spread a pair by inflating its noise rather than by moving it.
/// Here training mode carries no gradient, so only that pass can move the
/// encoder.
#[test]
fn the_penalty_reads_the_evaluation_mode_latent() {
    use super::PairPenalty;
    use crate::candle::decoder::gaussian_nb::GaussianNbDecoder;
    use crate::candle::vae::topic::{train_mixed_with_pairs, TrainConfig};
    use candle_core::DType;
    use candle_nn::{VarBuilder, VarMap};
    use nalgebra::DMatrix;
    use std::sync::atomic::AtomicBool;

    let (n, d, k) = (8usize, 4usize, 2usize);
    let x = DMatrix::<f32>::from_fn(n, d, |i, j| (1 + (i * 3 + j) % 5) as f32);
    let dev = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let w0 = vb
        .pp("enc")
        .get_with_hints((k, d), "weight", candle_nn::Init::Const(0.1))
        .unwrap();
    let mut enc = LinearEncoder {
        lin: candle_nn::Linear::new(w0.clone(), None),
        k,
        blind_in_training: true,
    };
    let before = v(&w0.flatten_all().unwrap());
    let dec = GaussianNbDecoder::new(d, k, vb.pp("dec")).unwrap();
    let stop = AtomicBool::new(false);
    let config = TrainConfig {
        parameters: &varmap,
        dev: &dev,
        epochs: 2,
        gpu_mem_fraction: None,
        minibatch_size: 4,
        learning_rate: 0.01,
        topic_smoothing: 0.0,
        grad_clip: 0.0,
        stop: &stop,
        loss_hook: None,
    };
    let pairs = [LevelPairs {
        pairs: vec![(0, 1, 1.0)],
        margin: 10.0,
    }];
    let penalty = PairPenalty {
        per_level: &pairs,
        lambda: 1.0,
        metric: PairMetric::Euclidean,
        batch: 1,
    };
    train_mixed_with_pairs(&[(&x, None, &x)], &mut enc, &[dec], &config, Some(&penalty)).unwrap();
    let after = v(&w0.flatten_all().unwrap());
    assert_ne!(before, after, "the penalty's pass reached the encoder");
}
