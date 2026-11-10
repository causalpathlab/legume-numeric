//! Numeric / ML foundation for the legume ecosystem.
//!
//! Modules:
//! - [`fast_math`]: fast, approximate `f32` `ln` and `ln Γ`, and `ln Γ` increments
//! - [`leiden`] — vendored CWTS/10x Leiden port (see `NOTICE`)
//! - [`matrix`] — matrix IO, KNN, layout, stats (ex `matrix-util`)
//! - `param` — parametric helpers (ex `matrix-param`)
//! - `candle` — candle training helpers (ex `candle-util`; feature `candle`)
//! - `mcmc` — MCMC engine: elliptical slice sampler, chain runner, diagnostics (feature `mcmc`)
//!
//! Features: `matrix` and `param` work on nalgebra `DMatrix` alone.
//! `tensor` (default) adds candle `Tensor` support to the matrix code:
//! conversions, tensor IO, sampling and fused ops. `ndarray` (default) adds
//! ndarray `Array2` support: IO, sampling, `ndarray_stat`, `ndarray_gamma`
//! and kNN from ndarray views. `candle` builds on `tensor`. `metal` and
//! `cuda` also run matrix code on the GPU (`Umap::fit_on`). A crate
//! that only needs `DMatrix` sets `default-features = false` and links
//! neither candle nor ndarray.

pub mod fast_math;
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
