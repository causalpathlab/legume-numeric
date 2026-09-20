//! Numeric / ML foundation for the legume ecosystem.
//!
//! Modules:
//! - [`leiden`] — vendored CWTS/10x Leiden port (see `NOTICE`)
//! - [`matrix`] — matrix IO, KNN, layout, stats (ex `matrix-util`)
//! - [`param`] — parametric helpers (ex `matrix-param`)
//! - [`candle`] — candle training helpers (ex `candle-util`; feature `candle`)
//! - [`mcmc`] — MCMC engine (ex `mcmc-util`; feature `mcmc`)

pub mod leiden;

#[cfg(feature = "matrix")]
pub mod matrix;

#[cfg(feature = "param")]
pub mod param;

#[cfg(feature = "candle")]
pub mod candle;

#[cfg(feature = "mcmc")]
pub mod mcmc;
