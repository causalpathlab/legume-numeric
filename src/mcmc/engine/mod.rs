//! Monte-Carlo accuracy diagnostics for a chain: [`ess`] (effective sample
//! size), [`split_rhat`] and [`mcse_proportion`].

pub mod diagnostics;

pub use diagnostics::{ess, mcse_proportion, split_rhat};
