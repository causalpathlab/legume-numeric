//! Tests for the peer-pair penalty: the latent distances, the hinge, the pair
//! batches a minibatch takes, the checks before training, and that it pushes
//! a close pair apart, from one gradient step to a whole training run.

use super::{check_pairs, latent_distance, pair_batch, pair_hinge, PairMetric};
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

/// Weighted mean of max(0, margin − d)²: pairs already apart cost nothing.
#[test]
fn hinge_charges_only_pairs_inside_the_margin() {
    let d = Tensor::new(&[0.0f32, 0.5, 2.0], &Device::Cpu).unwrap();
    let w = Tensor::new(&[1.0f32, 2.0, 1.0], &Device::Cpu).unwrap();
    let h = pair_hinge(&d, &w, 1.0).unwrap().to_scalar::<f32>().unwrap();
    // (1·1² + 2·0.5² + 1·0) / (1 + 2 + 1)
    assert!((h - 1.5 / 4.0).abs() < 1e-6, "{h}");
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

/// Pairs are checked once, before training: rows in range, and weights finite
/// and positive (a zero-weight batch would divide by zero, a negative weight
/// would pull the pair together).
#[test]
fn pairs_are_checked_before_training() {
    let ok = [(0, 1, 1.0), (2, 3, 0.5)];
    assert!(check_pairs(&ok, 4).is_ok());
    assert!(
        check_pairs(&ok, 3).is_err(),
        "row 3 is out of range for 3 rows"
    );
    for bad in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(
            check_pairs(&[(0, 1, bad)], 4).is_err(),
            "weight {bad} should be refused"
        );
    }
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
}

impl crate::candle::traits::model::EncoderModuleT for LinearEncoder {
    fn forward_t(
        &self,
        x_nd: &Tensor,
        _x0_nd: Option<&Tensor>,
        _train: bool,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        use candle_nn::Module;
        let z = self.lin.forward(&(x_nd + 1.0)?.log()?)?;
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
    use super::{LevelPairs, PairPenalty};
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
        let pairs: [LevelPairs; 1] = [vec![(0, 1, 1.0)]];
        let penalty = PairPenalty {
            per_level: &pairs,
            lambda: 50.0,
            margin: 2.0,
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
