use super::*;
use candle_core::Device;

fn layer(varmap: &VarMap) -> BatchNorm {
    let vb = VarBuilder::from_varmap(varmap, DType::F32, &Device::Cpu);
    batch_norm(3, BatchNormConfig::default(), varmap, vb.pp("bn")).unwrap()
}

fn rows(n: usize) -> Tensor {
    let v: Vec<f32> = (0..n * 3).map(|i| (i as f32).sin() * 2.0 + 1.0).collect();
    Tensor::from_vec(v, (n, 3), &Device::Cpu).unwrap()
}

fn all_finite(t: &Tensor) -> bool {
    t.flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .all(|v| v.is_finite())
}

/// An epoch of 2 × 4 + 1 rows ends on a one-row step. That step must not
/// poison the running statistics: evaluation afterwards stays finite.
#[test]
fn a_one_row_training_step_leaves_evaluation_finite() {
    let varmap = VarMap::new();
    let bn = layer(&varmap);
    for n in [4, 4, 1] {
        let y = bn.forward_t(&rows(n), true).unwrap();
        assert!(all_finite(&y), "training output of a {n}-row step");
    }
    assert!(all_finite(bn.running_mean.as_tensor()));
    assert!(all_finite(bn.running_var.as_tensor()));
    assert!(all_finite(&bn.forward_t(&rows(5), false).unwrap()));
}

/// The one-row step reads the running statistics and leaves them as they
/// were; a two-row step still updates them.
#[test]
fn only_a_batch_with_spread_updates_the_running_statistics() {
    let varmap = VarMap::new();
    let bn = layer(&varmap);
    bn.forward_t(&rows(4), true).unwrap();
    let before = bn.running_var.as_tensor().to_vec1::<f32>().unwrap();
    let one = bn.forward_t(&rows(1), true).unwrap();
    let eval = bn.forward_t(&rows(1), false).unwrap();
    assert_eq!(bn.running_var.as_tensor().to_vec1::<f32>().unwrap(), before);
    assert_eq!(
        one.to_vec2::<f32>().unwrap(),
        eval.to_vec2::<f32>().unwrap()
    );
    bn.forward_t(&rows(2), true).unwrap();
    assert_ne!(bn.running_var.as_tensor().to_vec1::<f32>().unwrap(), before);
}
