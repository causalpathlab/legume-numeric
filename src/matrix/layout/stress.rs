//! Stress majorization of a graph by stochastic gradient descent (Zheng,
//! Pawar and Goodman, 2018, "Graph drawing by stochastic gradient
//! descent", IEEE TVCG).
//!
//! Each edge of weight `w` (an affinity, bigger is closer) gets length
//! `1/w`, and the target distance between two nodes is the length of the
//! shortest path between them, found by Dijkstra from every node. The
//! layout minimizes the stress `sum_ij w_ij (|x_i - x_j| - d_ij)^2` with
//! `w_ij = d_ij^-2`, visiting every pair in a seeded random order each
//! iteration and moving its two nodes toward their target distance by a
//! step that anneals over 30 iterations from `1/w_min` to `0.1/w_max`, as
//! in the paper. It starts from classical MDS of the shortest-path lengths,
//! which a random start falls short of on a weighted path; nodes that the
//! start puts on one point (twins, alike to the MDS) are first nudged apart
//! by a tiny seeded amount, so the descent has a direction between them.
//! The graph is prepared as in [`PreparedGraph::new`]: edges of weight not
//! finite or not above zero, and self-loops, are dropped, and repeated
//! edges summed.
//!
//! A graph in separate pieces has no distance between its pieces, so each
//! piece is laid out on its own and the pieces are placed apart by
//! [`lay_out_by_piece`].
//!
//! Cost, for a piece of `m` nodes and `e` edges: `O(m e log m)` for the
//! shortest paths (in parallel over sources), `O(m^2)` memory for their
//! table and the pair list, and `O(m^2)` for the MDS start and for each of
//! the 30 descent iterations.

use super::graph_prep::{lay_out_by_piece, PreparedGraph};
use super::{classical_mds_2d, Mat};
use rand::{rngs::SmallRng, seq::SliceRandom, RngExt, SeedableRng};
use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

const ITERATIONS: usize = 30;
const EPSILON: f32 = 0.1;
/// How far apart, as a share of the shortest target distance, nodes that
/// the MDS start puts on one point are nudged.
const NUDGE: f32 = 1e-3;

/// 2D stress layout of the `n` nodes of an undirected graph given as
/// `(i, j, weight)` edges, weights being affinities. Nodes with no edges and
/// separate pieces are laid out apart (see the module doc). The same `seed`
/// gives the same layout. Panics if an edge end is not below `n`.
#[must_use]
pub fn stress_layout_2d_on_graph(n: usize, edges: &[(usize, usize, f32)], seed: u64) -> Mat {
    let graph = PreparedGraph::new(n, edges);
    lay_out_by_piece(&graph, |piece, _| piece_layout(piece, seed), None)
}

/// Stress layout of one connected piece, `m x 2`.
fn piece_layout(graph: &PreparedGraph, seed: u64) -> Mat {
    let m = graph.n();
    let d = shortest_paths(graph);
    // Each pair with its target distance, so the descent reads only this.
    let pairs: Vec<(u32, u32, f32)> = (0..m)
        .flat_map(|i| (0..i).map(move |j| (i, j)))
        .map(|(i, j)| (i as u32, j as u32, d[i * m + j]))
        .collect();
    let (d_min, d_max) = pairs
        .iter()
        .fold((f32::INFINITY, 0.0_f32), |(lo, hi), &(_, _, v)| {
            (lo.min(v), hi.max(v))
        });

    // Read row by row: the table is symmetric only up to round-off (a path
    // summed from either end), and the start is sensitive to that.
    let start = classical_mds_2d(&Mat::from_fn(m, m, |i, j| d[i * m + j] * d[i * m + j]));
    drop(d);
    let mut x: Vec<[f32; 2]> = (0..m).map(|i| [start[(i, 0)], start[(i, 1)]]).collect();
    let mut rng = SmallRng::seed_from_u64(seed);
    nudge_apart(&mut x, NUDGE * d_min, &mut rng);

    let eta_max = d_max * d_max;
    let eta_min = EPSILON * d_min * d_min;
    let lambda = (eta_max / eta_min).ln() / (ITERATIONS - 1) as f32;
    let mut order = pairs;
    for t in 0..ITERATIONS {
        let eta = eta_max * (-lambda * t as f32).exp();
        order.shuffle(&mut rng);
        for &(i, j, dij) in &order {
            let (i, j) = (i as usize, j as usize);
            let mu = (eta / (dij * dij)).min(1.0);
            let (dx, dy) = (x[i][0] - x[j][0], x[i][1] - x[j][1]);
            let mag = dx.hypot(dy);
            // A safety check only: the nudge parts the start's twins.
            if mag == 0.0 {
                continue;
            }
            let s = mu * 0.5 * (mag - dij) / mag;
            x[i][0] -= s * dx;
            x[i][1] -= s * dy;
            x[j][0] += s * dx;
            x[j][1] += s * dy;
        }
    }
    Mat::from_fn(m, 2, |i, c| x[i][c])
}

/// Move each node that sits within `step` of an earlier one by `step` in a
/// direction drawn from `rng`. Draws nothing when no two nodes are that
/// close, so other layouts keep their random sequence.
fn nudge_apart(x: &mut [[f32; 2]], step: f32, rng: &mut SmallRng) {
    for i in 1..x.len() {
        let close = (0..i).any(|j| (x[i][0] - x[j][0]).hypot(x[i][1] - x[j][1]) < step);
        if close {
            let theta = rng.random_range(0.0..std::f32::consts::TAU);
            x[i][0] += step * theta.cos();
            x[i][1] += step * theta.sin();
        }
    }
}

/// All-pairs shortest path lengths of a connected graph, each edge of
/// weight `w` of length `1/w`, as a row-major `m x m` table, by Dijkstra
/// from each node in parallel.
fn shortest_paths(graph: &PreparedGraph) -> Vec<f32> {
    let m = graph.n();
    let mut table = vec![f32::INFINITY; m * m];
    table.par_chunks_mut(m).enumerate().for_each(|(s, dist)| {
        let mut heap = BinaryHeap::new();
        dist[s] = 0.0;
        // Path lengths are non-negative and not NaN, and for such floats
        // the order of the bit patterns is the numeric order.
        heap.push(Reverse((0.0_f32.to_bits(), s)));
        while let Some(Reverse((du, u))) = heap.pop() {
            let du = f32::from_bits(du);
            for &(v, w) in graph.neighbours(u) {
                let dv = du + 1.0 / w;
                if dv < dist[v] {
                    dist[v] = dv;
                    heap.push(Reverse((dv.to_bits(), v)));
                }
            }
        }
    });
    table
}

#[cfg(test)]
mod tests;
