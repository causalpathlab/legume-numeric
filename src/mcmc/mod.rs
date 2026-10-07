//! MCMC engine: the elliptical slice sampler, a generic chain runner and
//! diagnostics (effective sample size, split R̂, Monte-Carlo error). Models
//! built on it, such as sparse regression, live in the crates that use them.

pub mod engine;
