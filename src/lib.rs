//! Numeric / ML foundation for the legume ecosystem.
//!
//! Modules:
//! - [`leiden`] — vendored CWTS/10x Leiden port (see `NOTICE`)
//! - [`matrix`] — matrix IO, KNN, layout, stats (ex `matrix-util`)
//! - `param` — parametric helpers (ex `matrix-param`)
//! - `candle` — candle training helpers (ex `candle-util`; feature `candle`)
//! - `mcmc` — MCMC diagnostics: effective sample size, split R̂ (feature `mcmc`)
//!
//! Features: `matrix` and `param` work on nalgebra `DMatrix` alone.
//! `tensor` (default) adds candle `Tensor` support to the matrix code:
//! conversions, tensor IO, sampling and fused ops. `ndarray` (default) adds
//! ndarray `Array2` support: IO, sampling, `ndarray_stat`, `ndarray_gamma`
//! and kNN from ndarray views. `candle` builds on `tensor`. A crate
//! that only needs `DMatrix` sets `default-features = false` and links
//! neither candle nor ndarray.

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
