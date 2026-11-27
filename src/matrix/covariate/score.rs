//! Gene-set scores of cells, block by block of a genes × cells count matrix.
//!
//! - A **module** set's score is Tirosh et al.'s (Science 352:189, 2016), as
//!   Seurat's `AddModuleScore` computes it: the set's mean log-normalised
//!   expression (`ln(1 + 10⁴ x / total)`) minus that of control genes drawn
//!   from the same expression bins (24 equal-size bins of the genes' mean,
//!   100 controls per set gene). It is a fixed linear functional of the
//!   cell's log expression: `1/|set|` on the set, `−1/|controls|` on the
//!   controls.
//! - A **share** set's score is `ln(1 + 10⁴ · share)`, the share being its
//!   genes' part of the cell's counts.
//!
//! Only the caller's `scored` rows take part: a cell's total is over them,
//! and controls are drawn from them, so a multiome caller scoring its RNA
//! rows keeps peak counts out of every share and every control. A set's
//! weights are recorded by row name ([`Functional`]), so new cells are
//! scored exactly as the fitted ones were; the caller maps names to its own
//! rows ([`ScoreWeights::new`]).

use nalgebra::DMatrix;
use nalgebra_sparse::CscMatrix;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

/// Expression bins controls are drawn from, and controls per set gene.
pub const N_BINS: usize = 24;
pub const N_CONTROLS: usize = 100;
/// Counts per cell the expression is normalised to.
pub const SCALE: f32 = 1e4;

/// How a set is scored.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ScoreKind {
    Module,
    Share,
}

/// One set's score as weights on named rows: on log expression (`module`)
/// or on counts (`share`, weight 1 on each set gene).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Functional {
    pub label: Box<str>,
    pub kind: ScoreKind,
    pub weights: Vec<(Box<str>, f32)>,
}

/// A set to score: its label, kind and rows.
pub struct SetRows<'a> {
    pub label: &'a str,
    pub kind: ScoreKind,
    pub rows: &'a [usize],
}

/// Each set's [`Functional`] over rows named `names`. Module sets need
/// `means`, each row's mean log-normalised expression ([`GeneMeans`]); their
/// controls come from the `scored` rows (all rows when `None`), drawn with a
/// stream seeded `seed`.
///
/// # Panics
/// A module set without `means`.
#[must_use]
pub fn functionals(
    sets: &[SetRows<'_>],
    names: &[Box<str>],
    means: Option<&[f64]>,
    scored: Option<&[usize]>,
    seed: u64,
) -> Vec<Functional> {
    let pool: Vec<usize> = scored.map_or_else(|| (0..names.len()).collect(), <[usize]>::to_vec);
    let bins = means.map(|m| ExpressionBins::new(m, &pool));
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    sets.iter()
        .map(|s| {
            let weights: Vec<(usize, f32)> = match s.kind {
                ScoreKind::Module => bins
                    .as_ref()
                    .expect("a module set needs the rows' means")
                    .weights(s.rows, &mut rng),
                ScoreKind::Share => s.rows.iter().map(|&g| (g, 1.0)).collect(),
            };
            Functional {
                label: s.label.into(),
                kind: s.kind,
                weights: weights
                    .into_iter()
                    .map(|(g, w)| (names[g].clone(), w))
                    .collect(),
            }
        })
        .collect()
}

/// Every functional's weights by row, on a caller's rows: what
/// [`ScoreWeights::score_block`] applies to a block, one pass over its
/// nonzeros.
pub struct ScoreWeights {
    /// Each row's `(functional, weight)` on log expression (module sets).
    log: Vec<Vec<(usize, f32)>>,
    /// Each row's `(functional, weight)` on counts (share sets).
    raw: Vec<Vec<(usize, f32)>>,
    share: Vec<bool>,
    scored: Option<Vec<bool>>,
}

impl ScoreWeights {
    /// The weights of `functionals` over `n_rows` rows, each named gene put on
    /// the row `row_of` finds for it (a gene it finds none for counts as
    /// unexpressed); `scored` as in [`functionals`].
    #[must_use]
    pub fn new(
        functionals: &[Functional],
        n_rows: usize,
        row_of: impl Fn(&str) -> Option<usize>,
        scored: Option<&[usize]>,
    ) -> Self {
        let (mut log, mut raw) = (vec![Vec::new(); n_rows], vec![Vec::new(); n_rows]);
        for (j, f) in functionals.iter().enumerate() {
            let by_row = match f.kind {
                ScoreKind::Module => &mut log,
                ScoreKind::Share => &mut raw,
            };
            for (g, x) in &f.weights {
                if let Some(r) = row_of(g) {
                    by_row[r].push((j, *x));
                }
            }
        }
        Self {
            log,
            raw,
            share: functionals
                .iter()
                .map(|f| f.kind == ScoreKind::Share)
                .collect(),
            scored: scored.map(|rows| mask(n_rows, rows)),
        }
    }

    /// The scores of a block of cells, cells × functionals, from its counts
    /// (rows × cells).
    #[must_use]
    pub fn score_block(&self, counts: &CscMatrix<f32>) -> DMatrix<f32> {
        let keep = |r: usize| self.scored.as_ref().is_none_or(|m| m[r]);
        let mut out = DMatrix::<f32>::zeros(counts.ncols(), self.share.len());
        for (i, col) in counts.col_iter().enumerate() {
            let entries = || {
                col.row_indices()
                    .iter()
                    .zip(col.values())
                    .filter(|(&r, _)| keep(r))
            };
            let total: f32 = entries().map(|(_, &v)| v).sum();
            if total <= 0.0 {
                continue;
            }
            let k = SCALE / total;
            for (&r, &v) in entries() {
                for &(j, w) in &self.raw[r] {
                    out[(i, j)] += w * v;
                }
                if !self.log[r].is_empty() {
                    let l = (v * k).ln_1p();
                    for &(j, w) in &self.log[r] {
                        out[(i, j)] += w * l;
                    }
                }
            }
            for (j, &share) in self.share.iter().enumerate() {
                if share {
                    out[(i, j)] = (k * out[(i, j)]).ln_1p();
                }
            }
        }
        out
    }
}

/// Each row's mean log-normalised expression, added up block by block (and
/// across threads with [`GeneMeans::merge`]).
pub struct GeneMeans {
    sum: Vec<f64>,
    cells: usize,
    scored: Option<Vec<bool>>,
}

impl GeneMeans {
    #[must_use]
    pub fn new(n_rows: usize, scored: Option<&[usize]>) -> Self {
        Self {
            sum: vec![0.0; n_rows],
            cells: 0,
            scored: scored.map(|rows| mask(n_rows, rows)),
        }
    }

    /// Add a block of cells' counts (rows × cells).
    pub fn add_block(&mut self, counts: &CscMatrix<f32>) {
        let keep = |r: usize| self.scored.as_ref().is_none_or(|m| m[r]);
        for col in counts.col_iter() {
            let entries = || {
                col.row_indices()
                    .iter()
                    .zip(col.values())
                    .filter(|(&r, _)| keep(r))
            };
            let total: f32 = entries().map(|(_, &v)| v).sum();
            if total > 0.0 {
                let k = SCALE / total;
                for (&r, &v) in entries() {
                    self.sum[r] += f64::from((v * k).ln_1p());
                }
            }
        }
        self.cells += counts.ncols();
    }

    pub fn merge(&mut self, other: &Self) {
        for (a, b) in self.sum.iter_mut().zip(&other.sum) {
            *a += b;
        }
        self.cells += other.cells;
    }

    #[must_use]
    pub fn means(&self) -> Vec<f64> {
        let n = self.cells.max(1) as f64;
        self.sum.iter().map(|s| s / n).collect()
    }
}

fn mask(n: usize, rows: &[usize]) -> Vec<bool> {
    let mut m = vec![false; n];
    for &r in rows {
        m[r] = true;
    }
    m
}

/// Rows in 24 equal-size bins of their mean expression, from which a module
/// score's controls are drawn (Seurat's `AddModuleScore`); what
/// [`functionals`] draws module sets' controls from.
pub struct ExpressionBins {
    bin: std::collections::HashMap<usize, usize>,
    members: Vec<Vec<usize>>,
}

impl ExpressionBins {
    /// The `pool` rows binned by `mean` (indexed by row), ties by row.
    #[must_use]
    pub fn new(mean: &[f64], pool: &[usize]) -> Self {
        let mut by_mean: Vec<usize> = pool.to_vec();
        by_mean.sort_by(|&a, &b| mean[a].total_cmp(&mean[b]).then(a.cmp(&b)));
        let d = by_mean.len().max(1);
        let mut bin = std::collections::HashMap::with_capacity(by_mean.len());
        let mut members: Vec<Vec<usize>> = vec![Vec::new(); N_BINS];
        for (rank, &g) in by_mean.iter().enumerate() {
            let b = rank * N_BINS / d;
            bin.insert(g, b);
            members[b].push(g);
        }
        Self { bin, members }
    }

    /// A module score's nonzero weights: `1/|set|` on the set, `−1/|controls|`
    /// on controls drawn from each set gene's bin (none for a gene outside
    /// the pool).
    #[must_use]
    pub fn weights(&self, set: &[usize], rng: &mut rand::rngs::SmallRng) -> Vec<(usize, f32)> {
        let mut controls: Vec<usize> = Vec::new();
        for g in set {
            if let Some(&b) = self.bin.get(g) {
                let pool = &self.members[b];
                let take = N_CONTROLS.min(pool.len());
                controls.extend(
                    rand::seq::index::sample(rng, pool.len(), take)
                        .into_iter()
                        .map(|i| pool[i]),
                );
            }
        }
        controls.sort_unstable();
        controls.dedup();
        let mut w: std::collections::BTreeMap<usize, f64> = std::collections::BTreeMap::new();
        for &g in &controls {
            *w.entry(g).or_default() -= 1.0 / controls.len() as f64;
        }
        for &g in set {
            *w.entry(g).or_default() += 1.0 / set.len() as f64;
        }
        w.into_iter()
            .filter(|(_, x)| *x != 0.0)
            .map(|(g, x)| (g, x as f32))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn module_weights_sum_to_zero_with_controls_from_the_set_s_bins() {
        let mean: Vec<f64> = (0..480u32).map(f64::from).collect();
        let pool: Vec<usize> = (0..480).collect();
        let set = [5usize, 300];
        let mut rng = rand::rngs::SmallRng::seed_from_u64(1);
        let w = ExpressionBins::new(&mean, &pool).weights(&set, &mut rng);
        assert!(w.iter().map(|&(_, x)| f64::from(x)).sum::<f64>().abs() < 1e-5);
        // Controls come from the set genes' bins (20 genes a bin here).
        for &(g, x) in &w {
            if x < 0.0 {
                assert!(g / 20 == 0 || g / 20 == 15, "gene {g}");
            }
        }
    }
}
