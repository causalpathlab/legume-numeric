//! Per-cell covariates (gene-set scores such as the cell cycle, mitochondrial
//! and ribosomal shares, sex or stress) binned as a second axis of batch,
//! shared by the tools that model batch by a per-gene δ: label each cell
//! `batch|cvK` and the δ is fitted per batch and bin, with no change to the
//! model.
//!
//! - [`score`]: module and share scores of a block of counts, recorded by
//!   row name so new cells are scored the same way.
//! - [`bins`]: the scores centred within batches, standardised robustly,
//!   judged per covariate and binned by one k-means (k by BIC), with the
//!   like-cell-group and small-bin guards; the fit assigns new cells.
//! - [`groups`]: the groups of like cells those guards use.
//!
//! Streaming the data, matching gene names and composing labels are the
//! caller's.

pub mod bins;
pub mod groups;
pub mod score;

pub use bins::{fit_bins, BinOpts, Binned, CovariateBins, NotBinned};
pub use groups::coarse_groups;
pub use score::{
    functionals, ExpressionBins, Functional, GeneMeans, ScoreKind, ScoreWeights, SetRows,
};
