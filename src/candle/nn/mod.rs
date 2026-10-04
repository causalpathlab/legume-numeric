//! Primitive neural-network building blocks shared across encoders,
//! decoders, and topic models.
//!
//! - [`linear`]: linear variants (incl. non-negative weights, indexed softmax)
//! - [`layers`]: activations, sparsemax, misc layer helpers
//! - [`batch_norm`](mod@batch_norm): VarMap-aware BatchNorm (device-transfer safe)

pub mod batch_norm;
pub mod layers;
pub mod linear;
pub mod seed_vars;
pub mod soft_clamp;

pub use batch_norm::{batch_norm, BatchNorm, BatchNormConfig};
pub use layers::{sparsemax, stack_relu_linear, StackLayers};
pub use linear::{
    aggregate_linear, aggregate_linear_hard, log_softmax_linear, log_softmax_linear_nobias,
    logsumexp_forward, non_neg_linear, sparsemax_linear, AggregateLinear, NonNegLinear,
    SoftmaxLinear, SparsemaxLinear,
};
pub use seed_vars::seed_uniform_vars;
pub use soft_clamp::{soft_clamp, MASKED_LOGIT_CLAMP};
