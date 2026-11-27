//! Cells binned by their covariate scores, as a second axis of batch.
//!
//! 1. Each score is centred by its batch's median: a shift between batches
//!    is the batch's own (its δ takes it), so only the spread within a
//!    batch is binned.
//! 2. Each is then standardised robustly ([`RobustScale`], clipped).
//! 3. With `judge` on, a covariate is binned only when its scores, taken
//!    together, show structure of their own (a mixture's BIC picks two
//!    components or more).
//! 4. One Gaussian mixture over the binned scores (diagonal covariances, EM
//!    from the best of several seeded k-means), its number of components by
//!    BIC up to `k_max` (or fixed); each cell in its most probable bin.
//! 5. Groups of like cells ([`super::coarse_groups`] of the random
//!    projection, the binned scores regressed out) at least `one_bin` in one
//!    bin keep no bin; within each batch a bin under `min_level` cells merges
//!    into its nearest, and a batch left with one bin keeps none.
//!
//! The fit ([`CovariateBins`]) is serialisable and assigns new cells the same
//! way ([`CovariateBins::assign`]): centred by their batch's recorded median
//! (a batch it has not seen, by its own), standardised with the recorded
//! scales, put in their most probable bin by the recorded mixture
//! ([`DiagGmm::assign`]), and moved as their batch's bins were.

use super::groups::coarse_groups;
use crate::matrix::gmm::{
    choose_gmm, fit_diag_gmm, kmeans_start, select_gmm, Criterion, DiagGmm, GmmChoice,
};
use crate::matrix::robust_band::{robust_scale_columns_inplace, RobustScale};
use crate::matrix::utils::median;
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// How cells are binned.
#[derive(Clone, Debug)]
pub struct BinOpts {
    /// Bin only covariates with structure of their own.
    pub judge: bool,
    /// Bins: fixed, else chosen by BIC up to `k_max`.
    pub fixed_k: Option<usize>,
    pub k_max: usize,
    /// What chooses the bins' number, and each covariate's own.
    pub criterion: Criterion,
    /// A group of like cells at least this share in one bin keeps no bin.
    pub one_bin: f32,
    /// A bin with fewer cells in a batch merges into its nearest.
    pub min_level: usize,
    /// Standardised scores are clipped at this many robust SDs.
    pub clip: f32,
    /// Seeded k-means starts per k (the mixture's start), the seed, and the
    /// EM steps at most.
    pub starts: usize,
    pub seed: u64,
    pub max_iter: usize,
    /// Groups of like cells, and the projection components they are
    /// clustered on.
    pub groups: usize,
    pub group_dims: usize,
}

impl Default for BinOpts {
    fn default() -> Self {
        Self {
            judge: true,
            fixed_k: None,
            k_max: 10,
            criterion: Criterion::Icl,
            one_bin: 0.95,
            min_level: 10,
            clip: 5.0,
            starts: 5,
            seed: 42,
            max_iter: 200,
            groups: 16,
            group_dims: 10,
        }
    }
}

/// A fitted binning, recorded beside a run.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CovariateBins {
    /// Each score's covariate (the input's columns, in order).
    pub covariate_of: Vec<Box<str>>,
    /// Each covariate's own number of bins by the criterion, over its scores
    /// together (judged covariates only).
    pub own_k: BTreeMap<Box<str>, usize>,
    /// The scores binned (input columns), and the covariates set aside.
    pub binned: Vec<usize>,
    pub set_aside: Vec<Box<str>>,
    /// Per batch, each binned score's median.
    pub batch_median: BTreeMap<Box<str>, Vec<f32>>,
    /// Each binned score's standardisation after centring.
    pub scales: Vec<RobustScale>,
    /// The criterion of each k tried (k = 1 first; empty when fixed), and
    /// the mixture in the standardised scores (its means the bins' centres;
    /// its components the bins).
    pub bic: Vec<f64>,
    pub mixture: DiagGmm,
    /// Per batch, where each bin's cells go: the bin, the one it merged
    /// into, or `None` (the batch keeps no bin).
    pub batch_bins: BTreeMap<Box<str>, Vec<Option<usize>>>,
    pub plain_cells: usize,
    pub one_bin_groups: usize,
}

/// Each cell's bin (`None`: no bin) and the fit.
pub type Binned = (Vec<Option<usize>>, CovariateBins);

/// Why nothing was binned.
#[derive(Debug, PartialEq)]
pub enum NotBinned {
    /// No score varies, or (judging) no covariate has structure of its own.
    NoStructure { set_aside: Vec<Box<str>> },
    /// The mixture chose one bin.
    OneBin { bic: Vec<f64> },
}

impl std::fmt::Display for NotBinned {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoStructure { set_aside } if set_aside.is_empty() => {
                write!(f, "no score varies")
            }
            Self::NoStructure { set_aside } => write!(
                f,
                "no covariate has structure of its own ({} set aside)",
                set_aside.join(", ")
            ),
            Self::OneBin { bic } => write!(f, "one bin (BIC by k: {bic:?})"),
        }
    }
}

/// Each cell's bin (`None`: no bin) and the fit, or why nothing was binned.
/// `scores` is cells × scores, `covariate_of` names each column's covariate
/// (a covariate's columns judged together), `batch` each cell's batch and
/// `proj_kn` the cells' random projection (dims × cells) for the groups of
/// like cells.
///
/// # Errors
/// Inputs of disagreeing lengths, or a projection the groups cannot be
/// made from.
pub fn fit_bins(
    scores: &DMatrix<f32>,
    covariate_of: &[Box<str>],
    batch: &[Box<str>],
    proj_kn: &DMatrix<f32>,
    o: &BinOpts,
) -> anyhow::Result<Result<Binned, NotBinned>> {
    let (n, m) = (scores.nrows(), scores.ncols());
    anyhow::ensure!(
        covariate_of.len() == m && batch.len() == n && proj_kn.ncols() == n,
        "scores, covariates, batches and projection disagree on their sizes"
    );
    anyhow::ensure!(
        o.fixed_k.is_none_or(|k| k >= 2) && o.k_max >= 1,
        "covariate bins: a fixed number of bins must be at least 2 and the most bins at least 1"
    );
    let by_batch = cells_by_batch(batch);
    // 1–2. Centred within batches, then standardised.
    let medians: BTreeMap<Box<str>, Vec<f32>> = by_batch
        .iter()
        .map(|(b, cells)| ((*b).into(), column_medians(scores, cells)))
        .collect();
    let mut z = scores.clone();
    for (b, cells) in &by_batch {
        let med = &medians[*b];
        for &i in cells {
            for j in 0..m {
                z[(i, j)] -= med[j];
            }
        }
    }
    let scales = robust_scale_columns_inplace(&mut z, o.clip);

    // 3. Each covariate judged on its varying scores together.
    let mut covariates: Vec<(&str, Vec<usize>)> = Vec::new();
    for (j, c) in covariate_of.iter().enumerate() {
        if covariates
            .last()
            .is_none_or(|(last, _)| *last != c.as_ref())
        {
            covariates.push((c, Vec::new()));
        }
        if scales[j].is_some() {
            covariates.last_mut().expect("pushed").1.push(j);
        }
    }
    covariates.retain(|(_, cols)| !cols.is_empty());
    // Judging, each covariate's own k on the rows it is chosen on (a
    // covariate binned alone keeps its choice for the joint mixture).
    let mut own_k = BTreeMap::new();
    let (mut binned, mut set_aside) = (Vec::new(), Vec::new());
    let mut alone: Option<GmmChoice> = None;
    for (c, cols) in &covariates {
        if !o.judge {
            binned.extend(cols);
            continue;
        }
        let choice = select_gmm(
            &z.select_columns(cols),
            o.k_max,
            o.criterion,
            o.starts,
            o.seed,
            o.max_iter,
        );
        let k = choice.fit.model.k();
        own_k.insert(Box::from(*c), k);
        if k >= 2 {
            alone = binned.is_empty().then_some(choice);
            binned.extend(cols);
        } else {
            set_aside.push(Box::from(*c));
        }
    }
    if binned.is_empty() {
        return Ok(Err(NotBinned::NoStructure { set_aside }));
    }
    let alone = alone.filter(|_| own_k.values().filter(|&&k| k >= 2).count() == 1);

    // 4. One mixture over the binned scores.
    let zb = z.select_columns(&binned);
    let (fit, bic) = match (o.fixed_k, alone) {
        (Some(k), _) => (
            fit_diag_gmm(&zb, &kmeans_start(&zb, k, o.starts, o.seed), o.max_iter),
            Vec::new(),
        ),
        (None, Some(c)) => {
            let c = c.on_all(&zb, o.max_iter);
            (c.fit, c.score)
        }
        (None, None) => {
            let c = choose_gmm(&zb, o.k_max, o.criterion, o.starts, o.seed, o.max_iter);
            (c.fit, c.score)
        }
    };
    let k = fit.model.k();
    if k < 2 {
        return Ok(Err(NotBinned::OneBin { bic }));
    }

    // 5. The guards.
    let mut bin: Vec<Option<usize>> = fit.labels.iter().map(|&c| Some(c)).collect();
    let remove: Vec<&[f32]> = zb.as_slice().chunks(n.max(1)).collect();
    let groups = coarse_groups(proj_kn, &remove, o.groups, o.group_dims, o.seed)?;
    let one_bin_groups = clear_one_bin_groups(&mut bin, &groups, k, o.one_bin);
    let centres = DMatrix::from_fn(k, binned.len(), |c, j| fit.model.means[c][j]);
    let batch_bins = merge_sparse_levels(&mut bin, &by_batch, &centres, o.min_level);
    let plain_cells = bin.iter().filter(|c| c.is_none()).count();
    let model = CovariateBins {
        covariate_of: covariate_of.to_vec(),
        own_k,
        batch_median: medians
            .into_iter()
            .map(|(b, med)| (b, binned.iter().map(|&j| med[j]).collect()))
            .collect(),
        scales: binned
            .iter()
            .map(|&j| scales[j].expect("binned scores vary"))
            .collect(),
        binned,
        set_aside,
        bic,
        mixture: fit.model,
        batch_bins,
        plain_cells,
        one_bin_groups,
    };
    Ok(Ok((bin, model)))
}

impl CovariateBins {
    /// New cells' bins: `scores` (cells × the input's scores, in the fit's
    /// column order) and their batches, centred, standardised and binned as
    /// the fitted cells were; see the module.
    #[must_use]
    pub fn assign(&self, scores: &DMatrix<f32>, batch: &[Box<str>]) -> Vec<Option<usize>> {
        let n = scores.nrows();
        let mut z = scores.select_columns(&self.binned);
        let by_batch = cells_by_batch(batch);
        for (b, cells) in &by_batch {
            let own;
            let med: &[f32] = match self.batch_median.get(*b) {
                Some(m) => m,
                None => {
                    own = column_medians(&z, cells);
                    &own
                }
            };
            for &i in cells {
                for (j, s) in self.scales.iter().enumerate() {
                    z[(i, j)] = s.apply(z[(i, j)] - med[j]);
                }
            }
        }
        let nearest = self.mixture.assign(&z);
        (0..n)
            .map(|i| match self.batch_bins.get(&batch[i]) {
                Some(to) => to.get(nearest[i]).copied().flatten(),
                None => Some(nearest[i]),
            })
            .collect()
    }
}

/// Each column's median over `cells`.
fn column_medians(x: &DMatrix<f32>, cells: &[usize]) -> Vec<f32> {
    (0..x.ncols())
        .map(|j| median(&cells.iter().map(|&i| x[(i, j)]).collect::<Vec<_>>()))
        .collect()
}

fn cells_by_batch(batch: &[Box<str>]) -> BTreeMap<&str, Vec<usize>> {
    let mut by: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (i, b) in batch.iter().enumerate() {
        by.entry(b.as_ref()).or_default().push(i);
    }
    by
}

/// Clear the bin of every cell in a group at least `one_bin` in one bin;
/// returns how many groups.
fn clear_one_bin_groups(
    bin: &mut [Option<usize>],
    groups: &[usize],
    k: usize,
    one_bin: f32,
) -> usize {
    let n_groups = groups.iter().copied().max().map_or(0, |g| g + 1);
    let mut count = vec![vec![0usize; k]; n_groups];
    for (c, &g) in bin.iter().zip(groups) {
        if let Some(c) = c {
            count[g][*c] += 1;
        }
    }
    let one: Vec<bool> = count
        .iter()
        .map(|c| {
            let total: usize = c.iter().sum();
            total > 0 && *c.iter().max().unwrap_or(&0) as f32 >= one_bin * total as f32
        })
        .collect();
    for (c, &g) in bin.iter_mut().zip(groups) {
        if one[g] {
            *c = None;
        }
    }
    one.iter().filter(|&&o| o).count()
}

/// Within each batch, merge a bin with fewer than `min_level` cells into the
/// bin whose centre is nearest, smallest first; a batch left with one bin
/// keeps none. Returns, per batch, where each bin goes: a bin the batch has
/// (its own, the one it merged into, or for a bin none of its cells had,
/// the nearest), or `None` for a batch that keeps no bin.
fn merge_sparse_levels(
    bin: &mut [Option<usize>],
    by_batch: &BTreeMap<&str, Vec<usize>>,
    centroids: &DMatrix<f32>,
    min_level: usize,
) -> BTreeMap<Box<str>, Vec<Option<usize>>> {
    let k = centroids.nrows();
    let dist = |a: usize, b: usize| (centroids.row(a) - centroids.row(b)).norm_squared();
    let mut remap = BTreeMap::new();
    for (b, cells) in by_batch {
        let mut to: Vec<Option<usize>> = (0..k).map(Some).collect();
        loop {
            let mut count: BTreeMap<usize, usize> = BTreeMap::new();
            for &i in cells {
                if let Some(c) = bin[i] {
                    *count.entry(c).or_default() += 1;
                }
            }
            if count.len() <= 1 {
                for &i in cells {
                    bin[i] = None;
                }
                break;
            }
            let Some((&small, &size)) = count.iter().min_by_key(|(_, &s)| s) else {
                break;
            };
            if size >= min_level {
                break;
            }
            let into = count
                .keys()
                .copied()
                .filter(|&c| c != small)
                .min_by(|&a, &c| dist(small, a).total_cmp(&dist(small, c)))
                .expect("another bin");
            for &i in cells {
                if bin[i] == Some(small) {
                    bin[i] = Some(into);
                }
            }
            for t in &mut to {
                if *t == Some(small) {
                    *t = Some(into);
                }
            }
        }
        let present: Vec<usize> = (0..k)
            .filter(|&c| cells.iter().any(|&i| bin[i] == Some(c)))
            .collect();
        for (c, t) in to.iter_mut().enumerate() {
            *t = match *t {
                Some(x) if present.contains(&x) => Some(x),
                _ => present
                    .iter()
                    .copied()
                    .min_by(|&a, &x| dist(c, a).total_cmp(&dist(c, x))),
            };
        }
        remap.insert(Box::from(*b), to);
    }
    remap
}

#[cfg(test)]
#[path = "bins_tests.rs"]
mod tests;
