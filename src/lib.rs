//! Numeric / ML foundation for the legume ecosystem.
//!
//! Modules:
//! - [`leiden`] — vendored CWTS/10x Leiden port (see `NOTICE`)
//! - [`matrix`] — matrix IO, KNN, layout, stats (ex `matrix-util`)
//! - [`param`] — parametric helpers (ex `matrix-param`)
//! - [`candle`] — candle training helpers (ex `candle-util`; feature `candle`)
//! - [`mcmc`] — MCMC engine (ex `mcmc-util`; feature `mcmc`)
//!
//! Features: `matrix` and `param` need no candle. `tensor` (default) adds
//! candle `Tensor` support to the matrix code: conversions, tensor IO,
//! sampling and fused ops. `candle` and `mcmc` build on `tensor`. A crate
//! that only needs matrices sets `default-features = false` and links no
//! candle at all.

pub mod leiden;

/// candle-core, for crates that use `Tensor` through the `tensor` feature
/// without a direct dependency on candle.
#[cfg(feature = "tensor")]
pub use candle_core;

#[cfg(feature = "matrix")]
pub mod matrix;

#[cfg(feature = "param")]
pub mod param;

#[cfg(feature = "candle")]
pub mod candle;

#[cfg(feature = "mcmc")]
pub mod mcmc;
