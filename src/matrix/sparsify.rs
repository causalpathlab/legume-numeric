//! Dynamics-preserving graph sparsification: a sparse subgraph whose linear
//! diffusion `ẋ = −Lx` follows the full graph's (Fitz-Gibbon, Jerdee, Zhang,
//! Braun & Moore 2026, "Dynamics-preserving graph sparsification",
//! arXiv:2611.23176).
//!
//! The error minimised is the Diffusion Spectral Error Φ, the expected
//! squared distance between the two graphs' diffusion trajectories over all
//! time from random starts ([`spectral`]). Unlike effective-resistance
//! sampling, which holds every Laplacian eigenvalue to the same relative
//! bound, Φ weighs each mode by its relaxation time 1/λ, so a small edge
//! budget goes to the slow modes the long-run dynamics are made of (and to
//! their eigenvectors). Bridges between modules are kept on purpose: this
//! preserves a graph's dynamics, it does not denoise it.
//!
//! The search ([`search`]) is exact, O(n³) per step (one eigendecomposition
//! of the sparse Laplacian), so it is for graphs of up to a few thousand
//! nodes — pseudobulk, landmark or cluster graphs. Larger pieces are
//! refused ([`DseArgs::max_nodes`]), never run slowly by surprise.
//!
//! On three-block graphs kept to 15–25% of their edges, Φ came out 3.5–4.3
//! times lower than effective-resistance sampling's where sampling gave a
//! connected graph at all; at 600 nodes and more none of 10 draws did.

mod search;
mod spectral;

use crate::matrix::graph::num_connected_components;
use crate::matrix::layout::graph_prep::{split, Piece, PreparedGraph};
use anyhow::{ensure, Result};
use nalgebra::SymmetricEigen;
use spectral::{laplacian, phi, Edge, Spectrum};

/// How [`dse_sparsify`] searches.
#[derive(Clone, Debug)]
pub struct DseArgs {
    /// Edges to keep in all; at least a spanning tree's, `n − pieces`.
    pub keep_edges: usize,
    /// Batches the edges past the spanning tree are added in (one
    /// eigendecomposition each).
    pub add_rounds: usize,
    /// Rounds of swaps then a weight fit.
    pub swap_rounds: usize,
    /// Swaps per round at most.
    pub max_swaps: usize,
    /// Swaps checked exactly (one eigendecomposition each) per swap made.
    pub exact_checks: usize,
    /// Steps of the weight fit per round.
    pub fit_steps: usize,
    /// The weight fit's step on log-weights.
    pub fit_rate: f64,
    /// The largest connected piece searched; a larger one is refused.
    pub max_nodes: usize,
}

impl DseArgs {
    /// Keep `keep_edges`, searching as the defaults say: fewer swaps and
    /// fit steps than the paper's (Algorithm 1: 50 swaps, 60 steps). On a
    /// 999-node block graph kept to a quarter of its edges, the paper's
    /// settings lowered Φ a further 1.3% at twice the time; the weight fit
    /// gives most of the gain (Φ 5.9 → 3.4 there), the swaps about 1%.
    /// About a minute at n ≈ 1000, a few seconds at n ≈ 300 (one
    /// eigendecomposition is 0.7 s at n = 999).
    #[must_use]
    pub fn keeping(keep_edges: usize) -> Self {
        Self {
            keep_edges,
            add_rounds: 16,
            swap_rounds: 2,
            max_swaps: 5,
            exact_checks: 1,
            fit_steps: 20,
            fit_rate: 0.1,
            max_nodes: 2000,
        }
    }
}

/// A sparse subgraph and how far its diffusion is from the full graph's.
#[derive(Clone, Debug)]
pub struct Sparsified {
    /// The kept edges `(i, j, w̃)`, `i < j`, with their new weights.
    pub edges: Vec<(usize, usize, f32)>,
    /// Φ, summed over the connected pieces.
    pub diffusion_error: f64,
}

/// The `n`-node graph `edges` (`(i, j, w)`) made clean as the graph layouts
/// make it ([`PreparedGraph`]: each pair once, repeats summed, self-loops
/// dropped), but a weight that is not positive and finite, or an end not
/// below `n`, refused rather than dropped: diffusion on a graph quietly
/// missing an edge is a different diffusion.
fn prepared(n: usize, edges: &[(usize, usize, f32)]) -> Result<PreparedGraph> {
    for &(i, j, w) in edges {
        ensure!(i < n && j < n, "edge ({i}, {j}) has an end not below {n}");
        ensure!(
            w.is_finite() && w > 0.0,
            "edge ({i}, {j}) has weight {w}: diffusion needs positive weights"
        );
    }
    Ok(PreparedGraph::new(n, edges))
}

/// A piece's edges in f64, the search's precision.
fn edges_f64(piece: &Piece) -> Vec<Edge> {
    piece
        .graph
        .edges()
        .iter()
        .map(|&(a, b, w)| (a, b, f64::from(w)))
        .collect()
}

/// Refuse a piece of `nodes` past `max_nodes`: each step is O(n³).
fn ensure_small(nodes: usize, max_nodes: usize) -> Result<()> {
    ensure!(
        nodes <= max_nodes,
        "a connected piece of {nodes} nodes is past max_nodes = {max_nodes}: each \
         eigendecomposition is O(n³)"
    );
    Ok(())
}

/// The sparse subgraph of the `n`-node graph `edges` with `args.keep_edges`
/// edges whose diffusion is closest to the full graph's (Φ). Each connected
/// piece is sparsified on its own, the budget shared in proportion to its
/// edges past its spanning tree (never below the tree). A graph already
/// within the budget is returned whole, unchecked; otherwise a piece that
/// has to be searched and is past `args.max_nodes` is refused.
pub fn dse_sparsify(n: usize, edges: &[(usize, usize, f32)], args: &DseArgs) -> Result<Sparsified> {
    let graph = prepared(n, edges)?;
    if graph.edges().len() <= args.keep_edges {
        return Ok(Sparsified {
            edges: graph.edges().to_vec(),
            diffusion_error: 0.0,
        });
    }
    let pieces = split(&graph);
    let tree_edges = n - pieces.len();
    ensure!(
        args.keep_edges >= tree_edges,
        "keeping {} edges cannot connect {n} nodes in {} pieces: at least {tree_edges} are needed",
        args.keep_edges,
        pieces.len()
    );
    let keeps = piece_budgets(&pieces, args.keep_edges - tree_edges);
    for (piece, &keep) in pieces.iter().zip(&keeps) {
        if keep < piece.graph.edges().len() {
            ensure_small(piece.nodes.len(), args.max_nodes)?;
        }
    }
    let mut out = Sparsified {
        edges: Vec::with_capacity(args.keep_edges),
        diffusion_error: 0.0,
    };
    for (piece, &keep) in pieces.iter().zip(&keeps) {
        let at = |a: usize, b: usize| (piece.nodes[a], piece.nodes[b]);
        if keep >= piece.graph.edges().len() {
            out.edges
                .extend(piece.graph.edges().iter().map(|&(a, b, w)| {
                    let (i, j) = at(a, b);
                    (i, j, w)
                }));
            continue;
        }
        let local = edges_f64(piece);
        let (weights, error) = search::Piece::new(piece.nodes.len(), &local, args).sparsify(keep);
        out.diffusion_error += error;
        out.edges
            .extend(local.iter().zip(&weights).filter(|(_, &w)| w > 0.0).map(
                |(&(a, b, _), &w)| {
                    let (i, j) = at(a, b);
                    (i, j, w as f32)
                },
            ));
    }
    out.edges.sort_by_key(|&(i, j, _)| (i, j));
    Ok(out)
}

/// Each piece's edges to keep: its spanning tree, and its share of the
/// `spare` edges past the trees, in proportion to its own edges past its
/// tree, what rounding down leaves going to the largest remainders. (With
/// fewer spare edges than the pieces have past their trees, which the
/// caller ensures, no share reaches its piece's edges and the leftover is
/// below the number of pieces with a remainder.)
fn piece_budgets(pieces: &[Piece], spare: usize) -> Vec<usize> {
    let extra: Vec<usize> = pieces
        .iter()
        .map(|p| p.graph.edges().len() + 1 - p.nodes.len())
        .collect();
    let total: usize = extra.iter().sum();
    let share: Vec<f64> = extra
        .iter()
        .map(|&x| spare as f64 * x as f64 / total as f64)
        .collect();
    let mut take: Vec<usize> = share.iter().map(|s| s.floor() as usize).collect();
    let mut by_remainder: Vec<usize> = (0..pieces.len()).collect();
    by_remainder
        .sort_by(|&a, &b| (share[b] - take[b] as f64).total_cmp(&(share[a] - take[a] as f64)));
    let left = spare - take.iter().sum::<usize>();
    for &p in by_remainder.iter().take(left) {
        take[p] += 1;
    }
    pieces
        .iter()
        .zip(take)
        .map(|(p, t)| p.nodes.len() - 1 + t)
        .collect()
}

/// Φ between the `n`-node graphs `full` and `sparse` (edges `(i, j, w)`),
/// summed over `full`'s connected pieces; infinite where `sparse` leaves a
/// piece disconnected (its diffusion never equilibrates there). Dense,
/// O(n³) in a piece's nodes: a piece past `max_nodes` is refused.
pub fn diffusion_spectral_error(
    n: usize,
    full: &[(usize, usize, f32)],
    sparse: &[(usize, usize, f32)],
    max_nodes: usize,
) -> Result<f64> {
    let pieces = split(&prepared(n, full)?);
    let sparse = prepared(n, sparse)?;
    let mut piece_of = vec![(0, 0); n];
    for (p, piece) in pieces.iter().enumerate() {
        if piece.nodes.len() > 1 {
            ensure_small(piece.nodes.len(), max_nodes)?;
        }
        for (local, &v) in piece.nodes.iter().enumerate() {
            piece_of[v] = (p, local);
        }
    }
    let mut sparse_in: Vec<Vec<Edge>> = vec![Vec::new(); pieces.len()];
    for &(i, j, w) in sparse.edges() {
        let ((p, a), (q, b)) = (piece_of[i], piece_of[j]);
        ensure!(
            p == q,
            "the sparse edge ({i}, {j}) joins two of the full graph's pieces"
        );
        sparse_in[p].push((a, b, f64::from(w)));
    }
    // Inside the full pieces, the sparse graph has as many pieces as they
    // only when it connects each of them.
    if num_connected_components(&sparse) > pieces.len() {
        return Ok(f64::INFINITY);
    }
    Ok(pieces
        .iter()
        .zip(&sparse_in)
        .filter(|(piece, _)| piece.nodes.len() > 1)
        .map(|(piece, sparse)| {
            let m = piece.nodes.len();
            phi(
                &Spectrum::of(m, &edges_f64(piece)),
                &Spectrum::of(m, sparse),
            )
        })
        .sum())
}

/// The normalized algebraic connectivity λ̂₂ of the `n`-node graph `edges`:
/// the second-smallest eigenvalue of `D^{−1/2} L D^{−1/2}`, in [0, 2]. Small
/// means a strong bottleneck (Cheeger: λ̂₂/2 ≤ conductance ≤ √(2λ̂₂)), where
/// DSE keeps diffusion far better than uniform spectral sparsifiers. 0 for
/// a disconnected graph; otherwise dense, O(n³), refused past `max_nodes`.
pub fn normalized_algebraic_connectivity(
    n: usize,
    edges: &[(usize, usize, f32)],
    max_nodes: usize,
) -> Result<f64> {
    let graph = prepared(n, edges)?;
    if n < 2 || num_connected_components(&graph) > 1 {
        return Ok(0.0);
    }
    ensure_small(n, max_nodes)?;
    let weighted: Vec<Edge> = graph
        .edges()
        .iter()
        .map(|&(a, b, w)| (a, b, f64::from(w)))
        .collect();
    let l = laplacian(n, &weighted);
    let scale: Vec<f64> = (0..n).map(|i| l[(i, i)].sqrt().recip()).collect();
    let normalized = nalgebra::DMatrix::from_fn(n, n, |i, j| l[(i, j)] * scale[i] * scale[j]);
    let mut values: Vec<f64> = SymmetricEigen::new(normalized)
        .eigenvalues
        .iter()
        .copied()
        .collect();
    values.sort_by(f64::total_cmp);
    Ok(values[1].max(0.0))
}

#[cfg(test)]
#[path = "sparsify/tests.rs"]
mod tests;
