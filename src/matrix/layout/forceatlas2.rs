//! ForceAtlas2 layout of a given weighted undirected graph (Jacomy,
//! Venturini, Heymann and Bastian, 2014, PLoS ONE 9(6): e98679), reduced
//! to the parts that its tests show matter.
//!
//! Every pair of nodes repels with force `1 / d`, and every edge pulls its
//! two ends together with force `w * d` (its weight times their distance).
//! Nodes move along their net force by the paper's adaptive global speed:
//! the ratio of total traction to total swinging (how much the forces on the
//! nodes keep or flip their direction between steps), times the paper's
//! `k_s`, with each move capped at the paper's `k_s_max` over the force. A
//! piece stops early once its forces no longer change. The start is a seeded
//! random placement, so a layout is deterministic for a given seed. The
//! graph is prepared as in [`PreparedGraph::new`]: edges of weight not
//! finite or not above zero, and self-loops, are dropped, and repeated
//! edges summed.
//!
//! Left out, each because breaking it changed no test: the degree weighting
//! of repulsion (masses `deg + 1`; on kNN-like graphs degrees are near
//! uniform), the paper's per-node speed (it only slowed convergence), the
//! cap on how fast the global speed may rise, and, with gravity gone, the
//! repulsion scaling `k_r`, which then only rescales the layout. LinLog
//! mode, overlap prevention and the Barnes-Hut approximation are out too.
//!
//! Gravity, which in the paper keeps disconnected parts together, is
//! replaced by laying out each piece (connected component; a node with no
//! edge is a piece of its own) on its own and then placing the pieces apart
//! by [`lay_out_by_piece`], in rows of a near-square shelf packing of their
//! bounding boxes, so no two overlap.
//!
//! Cost: exact repulsion over all pairs, so `O(n^2 + |E|)` time per
//! iteration (parallel over nodes) and `O(n + |E|)` memory.

use super::graph_prep::{lay_out_by_piece, PreparedGraph};
use super::Mat;
use rand::{rngs::SmallRng, RngExt, SeedableRng};
use rayon::prelude::*;

/// Speed constant `k_s`, scaling the global speed into a step.
const K_S: f32 = 0.1;
/// Cap on a node's displacement in one step, `k_s_max`.
const K_S_MAX: f32 = 10.0;

/// 2D ForceAtlas2 layout of the `n` nodes of the undirected graph `edges`
/// (`(i, j, weight)`, weights are affinities: larger pulls closer), run for
/// `iterations` steps from a start drawn with `seed`. Returns `n x 2`.
/// Panics if an edge end is not below `n`.
#[must_use]
pub fn forceatlas2_layout_2d_on_graph(
    n: usize,
    edges: &[(usize, usize, f32)],
    iterations: usize,
    seed: u64,
) -> Mat {
    let graph = PreparedGraph::new(n, edges);
    lay_out_by_piece(
        &graph,
        |piece, _| layout_piece(piece, iterations, seed),
        None,
    )
}

/// ForceAtlas2 on one connected piece, `m x 2`.
fn layout_piece(graph: &PreparedGraph, iterations: usize, seed: u64) -> Mat {
    let m = graph.n();
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut pos: Vec<[f32; 2]> = (0..m)
        .map(|_| [rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)])
        .collect();
    let mut old = vec![[0.0_f32; 2]; m];

    for _ in 0..iterations {
        let mut force: Vec<[f32; 2]> = (0..m)
            .into_par_iter()
            .map(|i| {
                let [xi, yi] = pos[i];
                let mut f = [0.0_f32; 2];
                for p in &pos {
                    let (dx, dy) = (xi - p[0], yi - p[1]);
                    let d2 = dx * dx + dy * dy;
                    if d2 > 0.0 {
                        let r = 1.0 / d2;
                        f[0] += r * dx;
                        f[1] += r * dy;
                    }
                }
                f
            })
            .collect();
        for &(i, j, w) in graph.edges() {
            let (dx, dy) = (pos[i][0] - pos[j][0], pos[i][1] - pos[j][1]);
            force[i][0] -= w * dx;
            force[i][1] -= w * dy;
            force[j][0] += w * dx;
            force[j][1] += w * dy;
        }

        let (mut total_swing, mut total_traction) = (0.0_f32, 0.0_f32);
        for (f, o) in force.iter().zip(&old) {
            total_swing += ((f[0] - o[0]).powi(2) + (f[1] - o[1]).powi(2)).sqrt();
            total_traction += ((f[0] + o[0]).powi(2) + (f[1] + o[1]).powi(2)).sqrt() / 2.0;
        }
        if total_swing == 0.0 {
            // At rest: forces did not change, and the speed would overflow.
            break;
        }
        let speed = total_traction / total_swing;

        for (p, f) in pos.iter_mut().zip(&force) {
            let norm = (f[0] * f[0] + f[1] * f[1]).sqrt();
            let s_i = (K_S * speed).min(K_S_MAX / norm);
            p[0] += s_i * f[0];
            p[1] += s_i * f[1];
        }
        old = force;
    }
    Mat::from_fn(m, 2, |i, c| pos[i][c])
}

#[cfg(test)]
mod tests;
