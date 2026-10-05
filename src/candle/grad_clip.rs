//! Global-norm gradient clipping for candle optimizers.
//!
//! candle's `Optimizer::backward_step` fuses `loss.backward()` + `step`, so
//! gradients flow into the optimizer unbounded. [`clipped_backward_step`]
//! restores the one-call ergonomics with a global-norm clip in between:
//! compute `‖g‖ = sqrt(Σ_i ‖g_i‖²)` over every parameter gradient and, when it
//! exceeds `max_norm`, scale them all by `max_norm / ‖g‖` — bounding the update
//! magnitude without changing its direction. Keeps embedding norms from
//! inflating on loss spikes.

use candle_core::backprop::GradStore;
use candle_core::{Result, Tensor};
use candle_nn::optim::Optimizer;

/// `loss.backward()` → global-norm clip → `opt.step()`, the clipped equivalent
/// of [`Optimizer::backward_step`]. `max_norm <= 0` disables clipping (plain
/// backward-step). Returns the pre-clip global gradient norm.
pub fn clipped_backward_step<O: Optimizer>(
    opt: &mut O,
    loss: &Tensor,
    max_norm: f64,
) -> Result<f64> {
    let mut grads = loss.backward()?;
    let norm = clip_grad_global_norm(&mut grads, max_norm)?;
    opt.step(&grads)?;
    Ok(norm)
}

/// `Σ ‖g‖²` over every gradient in `grads`, the same whatever order the store
/// yields them in. The store is a hash map, so its order changes between
/// runs, and a float sum taken in that order changes in its last bits; a
/// clip by it then steers two runs of one seed apart. The per-parameter
/// sums are stacked on the device, read in one host sync, and added in
/// sorted order in `f64`.
pub(crate) fn global_sumsq(grads: &GradStore) -> Result<f64> {
    let parts: Vec<Tensor> = grads
        .get_ids()
        .filter_map(|id| grads.get_id(*id))
        .map(|g| g.sqr()?.sum_all()?.to_dtype(candle_core::DType::F32))
        .collect::<Result<_>>()?;
    if parts.is_empty() {
        return Ok(0.0);
    }
    let mut sums: Vec<f32> = Tensor::stack(&parts, 0)?.to_vec1()?;
    sums.sort_by(f32::total_cmp);
    Ok(sums.iter().map(|&s| f64::from(s)).sum())
}

/// Clip every gradient in `grads` to a global L2 norm of `max_norm`. No-op when
/// `max_norm <= 0` (clipping disabled) or the global norm is already within
/// bound. Returns the pre-clip global norm (for logging / diagnostics).
///
/// One host read of the per-parameter sums of squares ([`global_sumsq`]).
pub fn clip_grad_global_norm(grads: &mut GradStore, max_norm: f64) -> Result<f64> {
    if max_norm <= 0.0 {
        return Ok(0.0);
    }
    // Release the immutable `get_ids` borrow before the mutable rescale pass.
    let ids: Vec<_> = grads.get_ids().copied().collect();
    let norm = global_sumsq(grads)?.sqrt();
    if norm > max_norm && norm > 0.0 {
        let scale = max_norm / norm;
        for id in &ids {
            if let Some(g) = grads.get_id(*id) {
                let scaled = g.affine(scale, 0.0)?;
                grads.insert_id(*id, scaled);
            }
        }
    }
    Ok(norm)
}
