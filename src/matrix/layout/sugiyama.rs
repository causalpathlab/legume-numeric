//! A layered (Sugiyama) drawing of the nodes of a given graph, top to
//! bottom from a root: for a trajectory, its course laid out as layers in
//! time and its branches side by side.
//!
//! 1. **Direction.** The layers are the breadth-first levels from the root.
//!    Every edge then joins two nodes of one level or of adjacent ones, so
//!    pointing each from the lower level to the higher gives an acyclic
//!    graph whose levels are a topological order, and no edge needs a dummy
//!    node to span layers. Without a given root, one end of the graph's
//!    longest path is taken (a double breadth-first sweep): on a trajectory
//!    whose progenitor stem is shorter than its branches that is a branch
//!    tip, which is why a caller who knows the root should give it.
//! 2. **Order within a layer.** Barycentre sweeps down the layers: each
//!    node moves to the mean place of its neighbours in the layer above,
//!    which keeps the edges from crossing and the branches apart.
//! 3. **Coordinates.** `y` is minus the level, `x` the place in the layer,
//!    centred. Separate pieces are placed apart by [`lay_out_by_piece`]; the
//!    one holding the root (else node 0) keeps its place, the root at the
//!    origin, and the others are shifted whole, so within each piece `y`
//!    still steps by one a level.
//!
//! The graph is prepared as in [`PreparedGraph::new`]: edges of weight not
//! finite or not above zero, and self-loops, are dropped.

use super::graph_prep::{lay_out_by_piece, PreparedGraph};
use super::Mat;
use crate::matrix::graph::{bfs, far_end};

/// Rounds of barycentre sweeps down the layers.
const SWEEPS: usize = 12;

/// The layered drawing of the `n` nodes of the undirected `edges`
/// (`(i, j, weight)`, weights otherwise unused), from `root` (else one end
/// of the longest path; a root not below `n` is ignored), each separate
/// piece from its own. Returns `n x 2`: `x` the place in the layer, `y`
/// minus the level. Panics if an edge end is not below `n`.
#[must_use]
pub fn sugiyama_layout_2d_on_graph(
    n: usize,
    edges: &[(usize, usize, f32)],
    root: Option<usize>,
) -> Mat {
    let graph = PreparedGraph::new(n, edges);
    let root = root.filter(|&r| r < n);
    let first = root.or((n > 0).then_some(0));
    lay_out_by_piece(
        &graph,
        |piece, at_first| {
            let from = match (root, at_first) {
                (Some(_), Some(r)) => r,
                _ => far_end(piece, far_end(piece, 0)),
            };
            layered(piece, from)
        },
        first,
    )
}

/// The layered drawing of one connected piece from `from`, `m x 2`.
fn layered(graph: &PreparedGraph, from: usize) -> Mat {
    let (order, _, level) = bfs(graph, from);
    let n_layers = order.last().map_or(0, |&v| level[v] + 1);
    let mut layers: Vec<Vec<usize>> = vec![Vec::new(); n_layers];
    for i in 0..graph.n() {
        layers[level[i]].push(i);
    }
    order_layers(graph, &mut layers);

    let mut xy = Mat::zeros(graph.n(), 2);
    for (l, ly) in layers.iter().enumerate() {
        let mid = (ly.len() as f32 - 1.0) / 2.0;
        for (k, &i) in ly.iter().enumerate() {
            xy[(i, 0)] = k as f32 - mid;
            xy[(i, 1)] = -(l as f32);
        }
    }
    xy
}

/// Barycentre sweeps down `layers` (each a list of nodes, in place order):
/// each node to the mean place of its neighbours in the layer above.
fn order_layers(graph: &PreparedGraph, layers: &mut [Vec<usize>]) {
    let mut place = vec![0.0_f32; graph.n()];
    let set_places = |ly: &[usize], place: &mut [f32]| {
        for (k, &i) in ly.iter().enumerate() {
            place[i] = k as f32;
        }
    };
    for ly in layers.iter() {
        set_places(ly, &mut place);
    }
    let mut in_layer = vec![usize::MAX; graph.n()];
    for (l, ly) in layers.iter().enumerate() {
        for &i in ly {
            in_layer[i] = l;
        }
    }
    let reorder = |ly: &mut Vec<usize>, above: usize, place: &mut [f32]| {
        let key: Vec<f32> = ly
            .iter()
            .map(|&i| {
                let there: Vec<f32> = graph
                    .neighbour_ids(i)
                    .filter(|&j| in_layer[j] == above)
                    .map(|j| place[j])
                    .collect();
                if there.is_empty() {
                    place[i]
                } else {
                    there.iter().sum::<f32>() / there.len() as f32
                }
            })
            .collect();
        let mut idx: Vec<usize> = (0..ly.len()).collect();
        idx.sort_by(|&a, &b| key[a].total_cmp(&key[b]));
        *ly = idx.iter().map(|&k| ly[k]).collect();
        set_places(ly, place);
    };
    for _ in 0..SWEEPS {
        for (l, ly) in layers.iter_mut().enumerate().skip(1) {
            reorder(ly, l - 1, &mut place);
        }
    }
}

#[cfg(test)]
mod tests;
