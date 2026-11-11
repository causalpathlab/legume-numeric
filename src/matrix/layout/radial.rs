//! Radial tree layout of a weighted graph.
//!
//! The graph is reduced to its maximum spanning tree by edge weight, the
//! backbone that minimum spanning tree methods of trajectory inference
//! (Monocle, Trapnell et al. 2014; TSCAN, Ji and Ji 2016) read lineages
//! from. Weights are affinities, so the tree keeps the strongest links. It is
//! found by Kruskal's algorithm (Kruskal, 1956) with a union-find, edges of
//! equal weight taken in order of their ends. The graph is prepared as in
//! [`PreparedGraph::new`]: edges of weight not finite or not above zero, and
//! self-loops, are dropped, and repeated edges summed.
//!
//! The tree is then drawn with the radial wedge layout of Eades (1992): the
//! root sits at the origin and every node at a radius equal to its depth in
//! the tree (in hops). Each node owns an angular wedge, the root the whole
//! circle; a node's wedge is split among its children in proportion to the
//! number of leaves under each, in the order the tree found them, and each
//! child sits at the middle of its own share. Wedges of nodes at one depth
//! never overlap, so no two nodes share a place, and a long arm of the graph
//! comes out as one straight spoke.
//!
//! The root is the one given, or else the centre of the tree: the node whose
//! farthest tree node is fewest hops away, the middle of a longest path,
//! found by two breadth-first sweeps.
//!
//! A graph in separate pieces is a forest. Each tree is laid out on its own
//! and the trees are placed apart by [`lay_out_by_piece`]; the first tree,
//! the one holding the given root or else the one holding node 0, stays
//! where its root is at the origin. A node with no edges is a tree of one.
//!
//! Cost: O(E log E) to sort the edges, near O(E) for the union-find, and
//! O(n) for the sweeps and the placement.

use super::graph_prep::{lay_out_by_piece, PreparedGraph};
use super::Mat;
use crate::matrix::graph::{bfs, far_end, AdjListGraph};
use petgraph::unionfind::UnionFind;
use std::f64::consts::TAU;

/// Lay out the `n` nodes of the undirected graph `edges` (`(i, j, w)`, `w`
/// an affinity, bigger is closer) as an `n x 2` radial drawing of its
/// maximum spanning tree. `root`, if given, is placed at the origin; else
/// the centre of the tree holding node 0 is. Panics if `root` or an edge
/// end is not below `n`.
#[must_use]
pub fn radial_tree_layout_2d_on_graph(
    n: usize,
    edges: &[(usize, usize, f32)],
    root: Option<usize>,
) -> Mat {
    let graph = PreparedGraph::new(n, edges);
    let first = root.or((n > 0).then_some(0));
    lay_out_by_piece(
        &graph,
        |piece, at_first| {
            let tree = maximum_spanning_tree(piece);
            let tree_root = match (root, at_first) {
                (Some(_), Some(r)) => r,
                _ => tree_centre(&tree),
            };
            place_tree(&tree, tree_root)
        },
        first,
    )
}

/// The maximum spanning tree of a connected graph (Kruskal), its edges
/// listed at each node in the order they joined the tree.
fn maximum_spanning_tree(graph: &PreparedGraph) -> AdjListGraph {
    let edges = graph.edges();
    let mut by_weight: Vec<usize> = (0..edges.len()).collect();
    by_weight.sort_by(|&a, &b| edges[b].2.total_cmp(&edges[a].2));
    let mut sets = UnionFind::<usize>::new(graph.n());
    let tree: Vec<(usize, usize, f32)> = by_weight
        .into_iter()
        .map(|e| edges[e])
        .filter(|&(i, j, _)| sets.union(i, j))
        .collect();
    AdjListGraph::from_edges(graph.n(), &tree)
}

/// The centre of a tree: the middle node of a longest path, found as the
/// farthest node `a` from node 0, then the farthest node `b` from `a`,
/// then half way back from `b` towards `a`.
fn tree_centre(tree: &AdjListGraph) -> usize {
    let a = far_end(tree, 0);
    let (order, parent, depth) = bfs(tree, a);
    let mut c = *order.last().expect("a is reached");
    for _ in 0..depth[c] / 2 {
        c = parent[c];
    }
    c
}

/// The radial wedge layout of the tree rooted at `root`, the root at the
/// origin: `n x 2` for a tree of `n` nodes.
fn place_tree(tree: &AdjListGraph, root: usize) -> Mat {
    let (order, parent, depth) = bfs(tree, root);
    let n = order.len();
    let mut out = Mat::zeros(n, 2);
    let mut children: Vec<Vec<usize>> = vec![vec![]; n];
    for &v in &order[1..] {
        children[parent[v]].push(v);
    }
    let mut leaves = vec![0usize; n];
    for &v in order.iter().rev() {
        leaves[v] = if children[v].is_empty() {
            1
        } else {
            children[v].iter().map(|&c| leaves[c]).sum()
        };
    }
    let mut lo = vec![0.0f64; n];
    let mut width = vec![0.0f64; n];
    width[root] = TAU;
    for &v in &order {
        let mut at = lo[v];
        for &c in &children[v] {
            lo[c] = at;
            width[c] = width[v] * leaves[c] as f64 / leaves[v] as f64;
            at += width[c];
        }
        if v != root {
            let theta = lo[v] + 0.5 * width[v];
            let r = depth[v] as f64;
            out[(v, 0)] = (r * theta.cos()) as f32;
            out[(v, 1)] = (r * theta.sin()) as f32;
        }
    }
    out
}

#[cfg(test)]
mod tests;
