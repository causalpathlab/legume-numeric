//! What the graph layouts share: their input graph made clean once
//! ([`PreparedGraph`]), its split into connected pieces ([`split`]), and
//! the placement of the pieces' layouts apart ([`lay_out_by_piece`]).

use super::Mat;
use crate::matrix::graph::{connected_components, WeightedGraph};
use rayon::prelude::*;

/// An undirected graph of `n` nodes made ready for a layout: every edge has
/// a finite weight above zero, joins two distinct nodes, and appears once,
/// as `(low, high)`, with the weights of its repeats summed; the edges are
/// sorted. Each node's neighbours, with the edges' weights, are sorted by
/// node.
pub(super) struct PreparedGraph {
    edges: Vec<(usize, usize, f32)>,
    neighbours: Vec<Vec<(usize, f32)>>,
}

impl PreparedGraph {
    /// Prepare the `n` nodes and `(i, j, weight)` edges of a layout's input:
    /// edges whose weight is not finite or not above zero are dropped, and
    /// so are self-loops; repeated edges are summed. Panics if an edge end
    /// is not below `n`.
    pub(super) fn new(n: usize, edges: &[(usize, usize, f32)]) -> Self {
        let mut kept: Vec<(usize, usize, f32)> = edges
            .iter()
            .map(|&(i, j, w)| {
                assert!(
                    i < n && j < n,
                    "edge ({i}, {j}) has an end not below the node count {n}"
                );
                (i.min(j), i.max(j), w)
            })
            .filter(|&(i, j, w)| i != j && w.is_finite() && w > 0.0)
            .collect();
        kept.sort_by_key(|&(i, j, _)| (i, j));
        kept.dedup_by(|next, kept| {
            let repeat = (next.0, next.1) == (kept.0, kept.1);
            if repeat {
                kept.2 += next.2;
            }
            repeat
        });
        Self::from_sorted(n, kept)
    }

    /// From edges that are already `(low, high)`, distinct, sorted and of
    /// weight above zero.
    fn from_sorted(n: usize, edges: Vec<(usize, usize, f32)>) -> Self {
        // In sorted order a node's lower neighbours come before its higher
        // ones, each in increasing order, so the lists come out sorted.
        let mut neighbours = vec![Vec::new(); n];
        for &(i, j, w) in &edges {
            neighbours[i].push((j, w));
            neighbours[j].push((i, w));
        }
        Self { edges, neighbours }
    }

    /// The number of nodes.
    pub(super) fn n(&self) -> usize {
        self.neighbours.len()
    }

    /// The edges, `(low, high, weight)`, sorted.
    pub(super) fn edges(&self) -> &[(usize, usize, f32)] {
        &self.edges
    }

    /// The neighbours of `i` with the weights of their edges, sorted by node.
    pub(super) fn neighbours(&self, i: usize) -> &[(usize, f32)] {
        &self.neighbours[i]
    }

    /// The neighbours of `i`, sorted.
    pub(super) fn neighbour_ids(&self, i: usize) -> impl Iterator<Item = usize> + '_ {
        self.neighbours[i].iter().map(|&(j, _)| j)
    }

    /// The subgraph on the nodes `i` with `keep[i]`, renumbered in order.
    pub(super) fn induced(&self, keep: &[bool]) -> Self {
        let mut at = vec![usize::MAX; self.n()];
        let mut m = 0;
        for i in (0..self.n()).filter(|&i| keep[i]) {
            at[i] = m;
            m += 1;
        }
        let edges = self
            .edges
            .iter()
            .filter(|&&(i, j, _)| keep[i] && keep[j])
            .map(|&(i, j, w)| (at[i], at[j], w))
            .collect();
        Self::from_sorted(m, edges)
    }
}

impl WeightedGraph for PreparedGraph {
    fn num_nodes(&self) -> usize {
        self.n()
    }

    fn num_edges(&self) -> usize {
        self.edges.len()
    }

    fn neighbors_with_weight<'a>(
        &'a self,
        node: usize,
    ) -> Box<dyn Iterator<Item = (usize, f32)> + 'a> {
        Box::new(self.neighbours[node].iter().copied())
    }
}

/// One connected piece of a graph: its nodes, in increasing order, and the
/// graph between them, whose node `k` is `nodes[k]`.
pub(super) struct Piece {
    pub(super) nodes: Vec<usize>,
    pub(super) graph: PreparedGraph,
}

/// The connected pieces of `graph`, in the order of their lowest node; a
/// node with no edges is a piece of its own.
pub(super) fn split(graph: &PreparedGraph) -> Vec<Piece> {
    let label = connected_components(graph);
    let n_pieces = label.iter().max().map_or(0, |&c| c + 1);
    let mut nodes: Vec<Vec<usize>> = vec![Vec::new(); n_pieces];
    let mut local = vec![0; graph.n()];
    for (i, &c) in label.iter().enumerate() {
        local[i] = nodes[c].len();
        nodes[c].push(i);
    }
    let mut edges: Vec<Vec<(usize, usize, f32)>> = vec![Vec::new(); n_pieces];
    for &(i, j, w) in graph.edges() {
        edges[label[i]].push((local[i], local[j], w));
    }
    nodes
        .into_iter()
        .zip(edges)
        .map(|(nodes, edges)| {
            let graph = PreparedGraph::from_sorted(nodes.len(), edges);
            Piece { nodes, graph }
        })
        .collect()
}

/// Lay `graph` out piece by piece, the pieces in parallel, and place them
/// apart. Returns `n x 2`.
///
/// `piece` lays out one connected piece, given its graph and, when the
/// piece holds `first`, that node's number in it, as `m x 2` for `m`
/// nodes. A piece of one node is not passed to it but sits at the origin.
/// The pieces' bounding boxes are then placed by [`pack`], the piece holding
/// `first` first and the rest in the order of their lowest node. With
/// `first` given, that piece keeps its own coordinates (so a root that its
/// layout puts at the origin stays there); without, the packing starts at
/// the origin. Panics if `first` is not below `n`.
pub(super) fn lay_out_by_piece(
    graph: &PreparedGraph,
    piece: impl Fn(&PreparedGraph, Option<usize>) -> Mat + Sync,
    first: Option<usize>,
) -> Mat {
    let n = graph.n();
    if let Some(f) = first {
        assert!(f < n, "node {f} is not below the node count {n}");
    }
    let mut pieces = split(graph);
    let holds = |p: &Piece, f: usize| p.nodes.binary_search(&f).ok();
    if let Some(f) = first {
        let at = pieces
            .iter()
            .position(|p| holds(p, f).is_some())
            .expect("every node is in a piece");
        pieces[..=at].rotate_right(1);
    }
    let layouts: Vec<Mat> = pieces
        .par_iter()
        .map(|p| {
            if p.nodes.len() == 1 {
                Mat::zeros(1, 2)
            } else {
                piece(&p.graph, first.and_then(|f| holds(p, f)))
            }
        })
        .collect();

    let boxes: Vec<[f32; 4]> = layouts.iter().map(bounding_box).collect();
    let mut offsets = pack(&boxes);
    if let (Some(_), Some(&[dx, dy])) = (first, offsets.first()) {
        for o in &mut offsets {
            o[0] -= dx;
            o[1] -= dy;
        }
    }
    let mut out = Mat::zeros(n, 2);
    for ((p, y), off) in pieces.iter().zip(&layouts).zip(&offsets) {
        for (k, &i) in p.nodes.iter().enumerate() {
            out[(i, 0)] = y[(k, 0)] + off[0];
            out[(i, 1)] = y[(k, 1)] + off[1];
        }
    }
    out
}

/// The bounding box `[x_lo, y_lo, x_hi, y_hi]` of the rows of `y` (`m x 2`).
pub(super) fn bounding_box(y: &Mat) -> [f32; 4] {
    y.row_iter().fold(
        [
            f32::INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        ],
        |b, p| {
            [
                b[0].min(p[0]),
                b[1].min(p[1]),
                b[2].max(p[0]),
                b[3].max(p[1]),
            ]
        },
    )
}

/// Offsets that place `boxes` (`[x_lo, y_lo, x_hi, y_hi]`), in order, in
/// rows of a shelf packing about as wide as it is tall, the first box's
/// lower corner at the origin, with a gap between boxes of a tenth of the
/// largest box's longer side (1 when every box is a point), so no two
/// overlap.
fn pack(boxes: &[[f32; 4]]) -> Vec<[f32; 2]> {
    let size = |b: &[f32; 4]| [b[2] - b[0], b[3] - b[1]];
    let largest = boxes
        .iter()
        .map(|b| size(b)[0].max(size(b)[1]))
        .fold(0.0, f32::max);
    let gap = if largest > 0.0 { 0.1 * largest } else { 1.0 };
    let area: f32 = boxes
        .iter()
        .map(|b| (size(b)[0] + gap) * (size(b)[1] + gap))
        .sum();
    let row_width = area.sqrt();

    let (mut x, mut y, mut row_height) = (0.0_f32, 0.0_f32, 0.0_f32);
    boxes
        .iter()
        .map(|b| {
            let [w, h] = size(b);
            if x > 0.0 && x + w > row_width {
                x = 0.0;
                y += row_height + gap;
                row_height = 0.0;
            }
            let offset = [x - b[0], y - b[1]];
            x += w + gap;
            row_height = row_height.max(h);
            offset
        })
        .collect()
}

#[cfg(test)]
mod tests;
