//! Modified locally linear embedding (MLLE) to 2D of the nodes of a given
//! graph: the graph a layout already shares (a kNN graph, or the graph
//! between pseudobulks that [`crate::matrix::coarsen::collapse`] makes of
//! it) gives each node its neighbours, and the nodes' features their
//! reconstruction weights. Cells are placed on the result afterwards, as
//! after PHATE.
//!
//! LLE writes each point as an affine combination of its neighbours, then
//! finds the 2D coordinates that the same weights rebuild best: the bottom
//! eigenvectors of `M = (I − W)ᵀ(I − W)` after the constant one (Roweis and
//! Saul, 2000). A single weight vector per point leaves the 2D solution
//! underdetermined and the layout distorted, so MLLE (Zhang and Wang, 2007)
//! builds several independent ones from the bottom of each neighbourhood's
//! local spectrum.
//!
//! - **Nodes with fewer than three neighbours** cannot anchor the weights:
//!   the graph is peeled to its 3-core, and the peeled nodes are placed
//!   afterwards, in reverse order, at the mean of their placed neighbours.
//! - **Each connected piece is laid out on its own.** The bottom of the
//!   spectrum of a graph in pieces is the pieces' indicator vectors, which
//!   squash each to a point. Each piece gets its own eigenproblem, scaled
//!   to its RMS radius in the data, placed by classical MDS of the pieces'
//!   centroids, spread out together until none overlaps another.
//!
//! Costs: `O(m³)` for each piece's dense eigensolve, as PHATE's MDS.

use super::graph_prep::{split, PreparedGraph};
use super::{classical_mds_2d, rows_as_contiguous, sq_dist, Mat};
use crate::matrix::utils::median;
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;

type Mat64 = DMatrix<f64>;

/// The embedding is two-dimensional.
const DIM: usize = 2;

/// 2D MLLE of the rows of `features` (`n × D`), each row's neighbours its
/// own in the undirected graph `edges` (`(i, j, weight)`, weights otherwise
/// unused; edges of weight not finite or not above zero, and self-loops,
/// are dropped). Returns `n × 2`. Panics if an edge end is not below `n`.
#[must_use]
pub fn lle_layout_2d_on_graph(features: &Mat, edges: &[(usize, usize, f32)]) -> Mat {
    let n = features.nrows();
    let graph = PreparedGraph::new(n, edges);

    // The 3-core: peel nodes with fewer than DIM + 1 neighbours left.
    let mut left: Vec<usize> = (0..n).map(|i| graph.neighbours(i).len()).collect();
    let mut in_core = vec![true; n];
    let mut peeled: Vec<usize> = Vec::new();
    let mut stack: Vec<usize> = (0..n).filter(|&i| left[i] <= DIM).collect();
    while let Some(i) = stack.pop() {
        if !in_core[i] {
            continue;
        }
        in_core[i] = false;
        peeled.push(i);
        for j in graph.neighbour_ids(i) {
            if in_core[j] {
                left[j] -= 1;
                if left[j] == DIM {
                    stack.push(j);
                }
            }
        }
    }
    let core: Vec<usize> = (0..n).filter(|&i| in_core[i]).collect();
    let rows = rows_as_contiguous(features);
    let core_rows: Vec<&[f32]> = core.iter().map(|&i| rows[i].as_slice()).collect();

    let mut xy = Mat::zeros(n, DIM);
    let core_xy = lay_out_pieces(&core_rows, &graph.induced(&in_core));
    for (p, &i) in core.iter().enumerate() {
        xy.row_mut(i).copy_from(&core_xy.row(p));
    }

    // Peeled nodes, last peeled first, at the mean of their placed
    // neighbours; one with none sits at the layout's centre.
    let mut placed = in_core;
    let centre = if core.is_empty() {
        nalgebra::RowDVector::zeros(DIM)
    } else {
        core_xy.row_mean()
    };
    for &i in peeled.iter().rev() {
        let mine: Vec<usize> = graph.neighbour_ids(i).filter(|&j| placed[j]).collect();
        let at_i = if mine.is_empty() {
            centre.clone()
        } else {
            mine.iter()
                .fold(nalgebra::RowDVector::zeros(DIM), |s, &j| s + xy.row(j))
                / mine.len() as f32
        };
        xy.row_mut(i).copy_from(&at_i);
        placed[i] = true;
    }
    xy
}

/// The 2D layout of each connected piece of `graph` over `rows`: each
/// piece's MLLE, the pieces in parallel, scaled to its RMS radius in the
/// data, placed by the classical MDS of the pieces' centroids, then spread
/// out together so that no two pieces overlap.
fn lay_out_pieces(rows: &[&[f32]], graph: &PreparedGraph) -> Mat {
    let pieces = split(graph);
    let n_pieces = pieces.len();

    let dims = rows.first().map_or(0, |r| r.len());
    let centroids: Vec<Vec<f32>> = pieces
        .iter()
        .map(|p| {
            let mut c = vec![0.0_f32; dims];
            for &i in &p.nodes {
                c.iter_mut().zip(rows[i]).for_each(|(a, &b)| *a += b);
            }
            c.iter_mut().for_each(|a| *a /= p.nodes.len() as f32);
            c
        })
        .collect();

    // Each piece on its own, centred, scaled to its RMS radius in the data.
    let layouts: Vec<Mat> = pieces
        .par_iter()
        .zip(&centroids)
        .map(|(p, centroid)| {
            let piece_rows: Vec<&[f32]> = p.nodes.iter().map(|&i| rows[i]).collect();
            let coords = mlle_coords(&piece_rows, &p.graph);
            let ss: f32 = piece_rows.iter().map(|r| sq_dist(r, centroid)).sum();
            let scale = (ss / p.nodes.len() as f32).sqrt() / (DIM as f32).sqrt();
            coords * scale
        })
        .collect();

    let offsets = if n_pieces > 1 {
        log::info!("LLE: {n_pieces} separate pieces, each laid out on its own");
        let d2 = Mat::from_fn(n_pieces, n_pieces, |a, b| {
            sq_dist(&centroids[a], &centroids[b])
        });
        let radius: Vec<f32> = layouts
            .iter()
            .map(|y| y.row_iter().map(|r| r.norm()).fold(0.0, f32::max))
            .collect();
        apart(classical_mds_2d(&d2), &radius)
    } else {
        Mat::zeros(n_pieces, DIM)
    };

    let mut xy = Mat::zeros(rows.len(), DIM);
    for (c, p) in pieces.iter().enumerate() {
        for (k, &i) in p.nodes.iter().enumerate() {
            for d in 0..DIM {
                xy[(i, d)] = offsets[(c, d)] + layouts[c][(k, d)];
            }
        }
    }
    xy
}

/// `centres` (`c × 2`) scaled up together, just enough that discs of
/// `radius` around them overlap no other: one factor for all, so the MDS
/// arrangement is kept. Centres that coincide cannot be parted by scaling
/// and are left as they are.
fn apart(centres: Mat, radius: &[f32]) -> Mat {
    let c = centres.nrows();
    let mut factor = 1.0_f32;
    for a in 0..c {
        for b in 0..a {
            let d = (centres.row(a) - centres.row(b)).norm();
            if d > 1e-6 {
                factor = factor.max((radius[a] + radius[b]) / d);
            }
        }
    }
    centres * factor
}

/// MLLE coordinates of one connected piece, its node `i` at `rows[i]`:
/// `m × 2`, each axis of unit variance.
fn mlle_coords(rows: &[&[f32]], graph: &PreparedGraph) -> Mat {
    let m = rows.len();
    let nbrs: Vec<Vec<usize>> = (0..m).map(|i| graph.neighbour_ids(i).collect()).collect();
    if m <= DIM + 1 || nbrs.iter().any(|nb| nb.len() <= DIM) {
        return Mat::zeros(m, DIM);
    }

    // Each neighbourhood's local spectrum, largest first.
    let local: Vec<(DVector<f64>, Mat64)> = (0..m)
        .into_par_iter()
        .map(|i| {
            let k = nbrs[i].len();
            let eig = local_gram(rows[i], &nbrs[i], |j| rows[j]).symmetric_eigen();
            let mut o: Vec<usize> = (0..k).collect();
            o.sort_by(|&a, &b| eig.eigenvalues[b].total_cmp(&eig.eigenvalues[a]));
            // Round-off below the top value's 1e-12 counts as zero.
            let floor = 1e-12 * eig.eigenvalues[o[0]].max(0.0);
            let vals = DVector::from_fn(k, |r, _| {
                let v = eig.eigenvalues[o[r]];
                if v > floor {
                    v
                } else {
                    0.0
                }
            });
            let vecs = Mat64::from_fn(k, k, |r, c| eig.eigenvectors[(r, o[c])]);
            (vals, vecs)
        })
        .collect();

    // η: the median share of each neighbourhood off its top two directions.
    let rho: Vec<f32> = local
        .iter()
        .map(|(vals, _)| {
            let head: f64 = vals.rows(0, DIM).sum();
            (vals.rows(DIM, vals.len() - DIM).sum() / head.max(f64::MIN_POSITIVE)) as f32
        })
        .collect();
    let eta = f64::from(median(&rho));

    let blocks: Vec<Mat64> = local
        .par_iter()
        .map(|(vals, vecs)| mlle_block(vals, vecs, eta))
        .collect();

    // M = Σᵢ Sᵢ Wᵢ Wᵢᵀ Sᵢᵀ, where Wᵢ stacks −1 for i over its block.
    let mut big = Mat64::zeros(m, m);
    for (i, w) in blocks.iter().enumerate() {
        let nb = &nbrs[i];
        let ww = w * w.transpose();
        let row_sums: Vec<f64> = (0..nb.len()).map(|a| w.row(a).sum()).collect();
        for a in 0..nb.len() {
            for b in 0..nb.len() {
                big[(nb[a], nb[b])] += ww[(a, b)];
            }
            big[(i, nb[a])] -= row_sums[a];
            big[(nb[a], i)] -= row_sums[a];
        }
        big[(i, i)] += w.ncols() as f64;
    }

    // The bottom eigenvectors after the constant one.
    let eig = big.symmetric_eigen();
    let mut o: Vec<usize> = (0..m).collect();
    o.sort_by(|&a, &b| eig.eigenvalues[a].total_cmp(&eig.eigenvalues[b]));
    let scale = (m as f64).sqrt();
    Mat::from_fn(m, DIM, |r, d| {
        (eig.eigenvectors[(r, o[d + 1])] * scale) as f32
    })
}

/// MLLE's weight block for one point: `k × s`, one column per weight
/// vector, each column summing to one (Zhang and Wang, 2007).
///
/// `s` is the most bottom directions whose share of the spectrum stays
/// under `eta`, at most `k − 2`. A Householder reflection takes those
/// directions to vectors whose columns sum to the same value, so blended
/// with the ridge weights they all sum to one.
fn mlle_block(vals: &DVector<f64>, vecs: &Mat64, eta: f64) -> Mat64 {
    let k = vals.len();
    let r = ridge(vals.sum());
    let ones = DVector::from_element(k, 1.0);
    // The ridge weights: (C + ridge·I)⁻¹ 1, normalised.
    let mut w_reg = vecs * DVector::from_fn(k, |c, _| vecs.column(c).sum() / (vals[c] + r));
    let total = w_reg.sum();
    w_reg /= total;

    // s: bottom directions whose sum over the rest stays under η.
    let mut s = 1;
    let mut tail = 0.0;
    let all: f64 = vals.sum();
    for t in 1..=k - DIM {
        tail += vals[k - t];
        let head = all - tail;
        if head > 0.0 && tail / head <= eta {
            s = t;
        } else {
            break;
        }
    }

    let v = vecs.columns(k - s, s).into_owned();
    let col_sums = v.transpose() * &ones;
    let alpha = col_sums.norm() / (s as f64).sqrt();
    let mut h = DVector::from_element(s, alpha) - &col_sums;
    let norm_h = h.norm();
    if norm_h < 1e-12 {
        h.fill(0.0);
    } else {
        h /= norm_h;
    }
    let reflected = &v - (&v * &h) * (2.0 * h.transpose());
    reflected + w_reg * DVector::from_element(s, 1.0 - alpha).transpose()
}

/// The ridge on a local Gram matrix of trace `trace`: `10⁻³ · trace`, as
/// Roweis and Saul's, so its solves stay conditioned when there are fewer
/// features than neighbours. MLLE draws its weights from the bottom of the
/// local spectrum, so the layout does not hang on its size.
fn ridge(trace: f64) -> f64 {
    (1e-3 * trace).max(1e-12)
}

/// `G Gᵀ` for `G` the neighbours' offsets from `x`, in `f64`.
fn local_gram<'a>(x: &[f32], around: &[usize], row: impl Fn(usize) -> &'a [f32]) -> Mat64 {
    let offsets: Vec<Vec<f64>> = around
        .iter()
        .map(|&j| {
            row(j)
                .iter()
                .zip(x)
                .map(|(&a, &b)| f64::from(a - b))
                .collect()
        })
        .collect();
    let k = around.len();
    Mat64::from_fn(k, k, |a, b| {
        offsets[a].iter().zip(&offsets[b]).map(|(p, q)| p * q).sum()
    })
}

#[cfg(test)]
mod tests;
