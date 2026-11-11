//! Diffusion map of a given weighted undirected graph (Coifman and Lafon,
//! "Diffusion maps", Applied and Computational Harmonic Analysis, 2006).
//!
//! The edges' weights are the affinity W (symmetric, repeated edges summed;
//! the graph is prepared as in [`PreparedGraph::new`], so edges of weight
//! not finite or not above zero, and self-loops, are dropped), with degrees
//! D. The symmetric normalized operator S = D^-1/2 W D^-1/2 has the same
//! eigenvalues as the random walk P = D^-1 W; its eigenvectors phi_k give
//! P's right eigenvectors psi_k = phi_k / sqrt(d). The top one is
//! constant (eigenvalue 1); the 2D coordinates are the next two, each scaled
//! by lambda_k^t, so that Euclidean distance approximates diffusion distance
//! at time t (t = 0 gives the unscaled eigenvectors).
//!
//! A graph in separate pieces has one eigenvalue 1 per piece, so each
//! connected piece is laid out on its own; the pieces are then placed apart
//! by [`lay_out_by_piece`], so none overlaps. Isolated nodes, and nodes
//! whose edges are all dropped, are pieces of one node.
//!
//! Cost: a dense eigendecomposition per piece, O(m^2) memory and O(m^3)
//! time for a piece of m nodes, which is fine for a few thousand nodes.

use super::graph_prep::{lay_out_by_piece, PreparedGraph};
use super::Mat;
use nalgebra::{DMatrix, SymmetricEigen};

/// 2D diffusion map of the `n` nodes of the undirected graph `edges`
/// (`(i, j, affinity)`), at diffusion time `t`. Returns `n x 2`. Panics if
/// an edge end is not below `n`.
#[must_use]
pub fn diffusion_map_2d_on_graph(n: usize, edges: &[(usize, usize, f32)], t: usize) -> Mat {
    let graph = PreparedGraph::new(n, edges);
    lay_out_by_piece(&graph, |piece, _| connected_diffusion_map(piece, t), None)
}

/// Diffusion map of one connected piece, `m x 2`.
fn connected_diffusion_map(graph: &PreparedGraph, t: usize) -> Mat {
    let m = graph.n();
    let mut w = DMatrix::<f64>::zeros(m, m);
    for &(i, j, a) in graph.edges() {
        w[(i, j)] += f64::from(a);
        w[(j, i)] += f64::from(a);
    }
    let d: Vec<f64> = (0..m).map(|i| w.row(i).sum()).collect();
    let mut out = Mat::zeros(m, 2);
    let inv_sqrt: Vec<f64> = d.iter().map(|x| x.sqrt().recip()).collect();
    for i in 0..m {
        for j in 0..m {
            w[(i, j)] *= inv_sqrt[i] * inv_sqrt[j];
        }
    }
    let eig = SymmetricEigen::new(w);
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| eig.eigenvalues[b].total_cmp(&eig.eigenvalues[a]));
    for (dim, &k) in order.iter().skip(1).take(2).enumerate() {
        let scale = eig.eigenvalues[k].powi(i32::try_from(t).unwrap_or(i32::MAX));
        for i in 0..m {
            out[(i, dim)] = (eig.eigenvectors[(i, k)] * inv_sqrt[i] * scale) as f32;
        }
    }
    out
}

#[cfg(test)]
mod tests;
