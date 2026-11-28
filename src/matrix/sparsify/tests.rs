use super::*;
use crate::matrix::graph::is_connected;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// A graph: its node count and edges `(i, j, w)`, `i < j`.
type Graph = (usize, Vec<(usize, usize, f32)>);

/// Three blocks of `size` nodes, joined inside with probability `p_in` and
/// across with `p_out`, weights uniform in [0.5, 1.5]; a path through all
/// the nodes keeps it connected.
fn blocks(size: usize, p_in: f64, p_out: f64, seed: u64) -> Graph {
    let n = 3 * size;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut edges = Vec::new();
    for i in 0..n {
        for j in i + 1..n {
            let p = if i / size == j / size { p_in } else { p_out };
            if j == i + 1 || rng.random::<f64>() < p {
                edges.push((i, j, rng.random_range(0.5..1.5)));
            }
        }
    }
    (n, edges)
}

fn as_f64(edges: &[(usize, usize, f32)]) -> Vec<Edge> {
    edges
        .iter()
        .map(|&(a, b, w)| (a, b, f64::from(w)))
        .collect()
}

fn error(full: &Graph, sparse: &[(usize, usize, f32)]) -> f64 {
    diffusion_spectral_error(full.0, &full.1, sparse, 1000).unwrap()
}

fn tree_of(g: &Graph) -> Vec<(usize, usize, f32)> {
    crate::matrix::graph::maximum_spanning_tree(g.0, &g.1)
        .into_iter()
        .map(|e| g.1[e])
        .collect()
}

#[test]
fn phi_is_the_trajectory_error_integrated_over_time() {
    let full = blocks(6, 0.8, 0.1, 1);
    // Every third edge dropped past a spanning tree, the rest reweighted.
    let tree = tree_of(&full);
    let mut sparse = tree.clone();
    for (k, &(a, b, w)) in full.1.iter().enumerate() {
        if k % 3 != 0 && !tree.iter().any(|&(i, j, _)| (i, j) == (a, b)) {
            sparse.push((a, b, 1.7 * w));
        }
    }
    let closed = error(&full, &sparse);
    assert!(error(&full, &full.1) < 1e-20);

    // ∫ ‖e^{−L̃t} − e^{−Lt}‖²_F dt (the expectation over x₀ ~ N(0, I)),
    // on a log grid; the zero modes match and cancel.
    let (s, t) = (
        Spectrum::of(full.0, &as_f64(&full.1)),
        Spectrum::of(full.0, &as_f64(&sparse)),
    );
    let propagator = |sp: &Spectrum, time: f64| {
        sp.weighted_vectors(|l| (-l * time).exp()) * sp.vectors.transpose()
    };
    let times: Vec<f64> = (0..=4000)
        .map(|k| 1e-4 * 10f64.powf(k as f64 * 8.0 / 4000.0))
        .collect();
    let err = |time: f64| (propagator(&t, time) - propagator(&s, time)).norm_squared();
    let mut integral = 0.0;
    for w in times.windows(2) {
        integral += 0.5 * (err(w[0]) + err(w[1])) * (w[1] - w[0]);
    }
    assert!(
        (integral - closed).abs() < 1e-3 * closed,
        "integrated {integral}, closed form {closed}"
    );
}

#[test]
fn the_gradient_matches_finite_differences_on_kept_and_left_out_edges() {
    let (n, edges) = blocks(5, 0.9, 0.15, 2);
    let edges = as_f64(&edges);
    let full = Spectrum::of(n, &edges);
    // Every other edge plus the path, so the sparse graph is connected.
    let kept: Vec<bool> = edges
        .iter()
        .enumerate()
        .map(|(k, &(a, b, _))| k % 2 == 0 || b == a + 1)
        .collect();
    let sparse_with = |e: usize, extra: f64| -> Vec<Edge> {
        edges
            .iter()
            .enumerate()
            .filter_map(|(k, &(a, b, w))| {
                let w = if kept[k] { w } else { 0.0 } + if k == e { extra } else { 0.0 };
                (w > 0.0).then_some((a, b, w))
            })
            .collect()
    };
    let g = spectral::Gradient::new(&full, &Spectrum::of(n, &sparse_with(usize::MAX, 0.0)));
    let h = 1e-6;
    let kept_edge = (0..edges.len()).find(|&k| kept[k]).unwrap();
    let left_out = (0..edges.len()).find(|&k| !kept[k]).unwrap();
    for e in [kept_edge, left_out] {
        let up = phi(&full, &Spectrum::of(n, &sparse_with(e, h)));
        let at = phi(&full, &Spectrum::of(n, &sparse_with(e, 0.0)));
        let numeric = (up - at) / h;
        let exact = g.at(&edges[e]);
        assert!(
            (numeric - exact).abs() < 1e-3 * exact.abs().max(1e-3),
            "edge {e}: numeric {numeric}, exact {exact}"
        );
    }
}

/// Effective-resistance sampling as the paper runs it: each edge kept
/// independently with probability min(1, a·w·R), `a` set so the expected
/// count is `keep`, its weight divided by that probability.
fn resistance_sample(full: &Graph, keep: usize, seed: u64) -> Vec<(usize, usize, f32)> {
    let y = Spectrum::of(full.0, &as_f64(&full.1)).weighted_vectors(|l| l.sqrt().recip());
    let score: Vec<f64> = full
        .1
        .iter()
        .map(|&(a, b, w)| f64::from(w) * (y.row(a) - y.row(b)).norm_squared())
        .collect();
    let expected = |a: f64| score.iter().map(|s| (a * s).min(1.0)).sum::<f64>();
    let (mut lo, mut hi) = (0.0, 1e6);
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if expected(mid) < keep as f64 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let mut rng = StdRng::seed_from_u64(seed);
    full.1
        .iter()
        .zip(&score)
        .filter_map(|(&(a, b, w), s)| {
            let p = (hi * s).min(1.0);
            (rng.random::<f64>() < p).then_some((a, b, (f64::from(w) / p) as f32))
        })
        .collect()
}

#[test]
fn dse_keeps_a_modular_graph_s_diffusion_better_than_resistance_sampling() {
    let full = blocks(15, 0.6, 0.03, 3);
    let keep = full.1.len() / 4;
    let out = dse_sparsify(full.0, &full.1, &DseArgs::keeping(keep)).unwrap();
    assert_eq!(out.edges.len(), keep);
    assert!(is_connected(
        full.0,
        out.edges.iter().map(|&(i, j, _)| (i, j))
    ));
    let dse = error(&full, &out.edges);
    // Φ reported is Φ of the weights returned.
    assert!((dse - out.diffusion_error).abs() < 1e-9 * dse.max(1.0));
    // Resistance sampling, over draws that come out connected.
    let sampled: Vec<f64> = (0..20)
        .map(|seed| error(&full, &resistance_sample(&full, keep, seed)))
        .filter(|e| e.is_finite())
        .collect();
    assert!(!sampled.is_empty());
    let mean = sampled.iter().sum::<f64>() / sampled.len() as f64;
    assert!(
        dse < mean,
        "DSE Φ {dse} vs resistance sampling's mean {mean}"
    );
    // And the search improves on its starting tree.
    assert!(dse < error(&full, &tree_of(&full)));
}

#[test]
fn the_budget_is_kept_exactly_across_pieces_and_each_stays_connected() {
    // Two copies of a graph side by side, unjoined.
    let (m, a) = blocks(4, 0.9, 0.2, 4);
    let both: Vec<(usize, usize, f32)> = a
        .iter()
        .copied()
        .chain(a.iter().map(|&(i, j, w)| (i + m, j + m, w)))
        .collect();
    let full = (2 * m, both);
    for keep in [full.1.len() / 2, 2 * (m - 1) + 1] {
        let out = dse_sparsify(full.0, &full.1, &DseArgs::keeping(keep)).unwrap();
        assert_eq!(out.edges.len(), keep);
        assert!(error(&full, &out.edges).is_finite());
    }
}

#[test]
fn only_a_piece_that_is_searched_is_held_to_max_nodes() {
    // A long path (a tree, never searched) beside a small dense piece.
    let mut edges: Vec<(usize, usize, f32)> = (0..9).map(|i| (i, i + 1, 1.0)).collect();
    let (_, dense) = blocks(2, 1.0, 1.0, 8);
    edges.extend(dense.iter().map(|&(i, j, w)| (i + 10, j + 10, w)));
    let mut args = DseArgs::keeping(9 + 6);
    args.max_nodes = 6;
    assert!(dse_sparsify(16, &edges, &args).is_ok());
    args.max_nodes = 5;
    let err = dse_sparsify(16, &edges, &args).unwrap_err().to_string();
    assert!(err.contains("max_nodes"), "{err}");
    assert!(diffusion_spectral_error(16, &edges, &edges, 6).is_err());
}

#[test]
fn a_budget_below_a_tree_or_a_bad_weight_is_refused_and_parallel_edges_sum() {
    let (n, edges) = blocks(4, 0.9, 0.2, 5);
    assert!(dse_sparsify(n, &edges, &DseArgs::keeping(n - 2)).is_err());
    let bad = [(0, 1, 1.0), (1, 2, -1.0), (0, 2, 1.0)];
    assert!(dse_sparsify(3, &bad, &DseArgs::keeping(2)).is_err());
    // Already within the budget: the graph whole, repeats summed.
    let out = dse_sparsify(
        3,
        &[(0, 1, 1.0), (1, 0, 2.0), (1, 2, 1.0)],
        &DseArgs::keeping(2),
    )
    .unwrap();
    assert_eq!(out.edges, vec![(0, 1, 3.0), (1, 2, 1.0)]);
    assert_eq!(out.diffusion_error, 0.0);
}

#[test]
fn a_bottleneck_lowers_the_normalized_algebraic_connectivity() {
    let lambda = |g: Graph| normalized_algebraic_connectivity(g.0, &g.1, 100).unwrap();
    let modular = lambda(blocks(10, 0.9, 0.01, 6));
    let mixed = lambda(blocks(10, 0.9, 0.9, 6));
    assert!(modular < mixed, "{modular} vs {mixed}");
    assert!((0.0..=2.0).contains(&mixed));
    // Disconnected: 0, known without a decomposition, whatever the size.
    let apart = [(0, 1, 1.0), (2, 3, 1.0)];
    assert_eq!(
        normalized_algebraic_connectivity(4, &apart, 1).unwrap(),
        0.0
    );
    let (n, edges) = blocks(10, 0.9, 0.9, 6);
    assert!(normalized_algebraic_connectivity(n, &edges, 10).is_err());
}
