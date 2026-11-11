use super::*;
use crate::matrix::agreement::spearman;
use crate::matrix::landmark_init::tests::{swiss_roll, unrolled};
use crate::matrix::test_support::fuzzy_edges;
use rand::{rngs::SmallRng, RngExt, SeedableRng};

type Edges = Vec<(usize, usize, f32)>;

/// A `w x h` lattice with unit weights, nodes numbered row by row from
/// `offset`, and each node's place on it.
fn lattice(w: usize, h: usize, offset: usize) -> (Edges, Vec<[f32; 2]>) {
    let mut edges = vec![];
    let mut place = vec![];
    for r in 0..h {
        for c in 0..w {
            let i = offset + r * w + c;
            place.push([c as f32, r as f32]);
            if c + 1 < w {
                edges.push((i, i + 1, 1.0));
            }
            if r + 1 < h {
                edges.push((i, i + w, 1.0));
            }
        }
    }
    (edges, place)
}

fn dist(y: &Mat, i: usize, j: usize) -> f32 {
    (y[(i, 0)] - y[(j, 0)]).hypot(y[(i, 1)] - y[(j, 1)])
}

/// Rank correlation of all pairwise distances among `nodes`, layout `y`
/// against the true places `place` (`place[k]` is the place of `nodes[k]`).
fn faithful(y: &Mat, nodes: &[usize], place: &[[f32; 2]]) -> f32 {
    let (mut truth, mut laid) = (vec![], vec![]);
    for a in 0..nodes.len() {
        for b in 0..a {
            truth.push((place[a][0] - place[b][0]).hypot(place[a][1] - place[b][1]));
            laid.push(dist(y, nodes[a], nodes[b]));
        }
    }
    spearman(&truth, &laid)
}

fn flat(y: &Mat) -> Vec<f32> {
    (0..y.nrows())
        .flat_map(|i| [y[(i, 0)], y[(i, 1)]])
        .collect()
}

// Bar: rank correlation of pairwise distances, layout vs lattice, >= 0.98.
#[test]
fn a_grid_is_laid_out_as_its_grid() {
    let (edges, place) = lattice(15, 10, 0);
    let y = stress_layout_2d_on_graph(150, &edges, 1);
    let nodes: Vec<usize> = (0..150).collect();
    let rho = faithful(&y, &nodes, &place);
    eprintln!("grid rank correlation {rho}");
    assert!(rho >= 0.98, "grid rank correlation {rho}");
}

// Bar: projected on its principal axis, a 40-node chain keeps its order
// exactly (|Spearman| with the node index = 1).
#[test]
fn a_chain_keeps_its_order() {
    let n = 40;
    let edges: Vec<_> = (0..n - 1).map(|i| (i, i + 1, 1.0)).collect();
    let y = stress_layout_2d_on_graph(n, &edges, 3);
    let (mx, my) = (y.column(0).mean(), y.column(1).mean());
    let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let (a, b) = (y[(i, 0)] - mx, y[(i, 1)] - my);
        sxx += a * a;
        sxy += a * b;
        syy += b * b;
    }
    let theta = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    let along: Vec<f32> = (0..n)
        .map(|i| (y[(i, 0)] - mx) * theta.cos() + (y[(i, 1)] - my) * theta.sin())
        .collect();
    let index: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let rho = spearman(&index, &along).abs();
    eprintln!("chain order {rho}");
    assert!(rho >= 0.9999, "chain order {rho}");
}

// Bar: each segment of a weighted path is 1/w long, within 2%.
#[test]
fn edge_lengths_follow_inverse_weights() {
    let weights = [1.0_f32, 0.25, 0.5, 2.0];
    let edges: Vec<_> = weights
        .iter()
        .enumerate()
        .map(|(i, &w)| (i, i + 1, w))
        .collect();
    let y = stress_layout_2d_on_graph(5, &edges, 5);
    for (i, &w) in weights.iter().enumerate() {
        let got = dist(&y, i, i + 1);
        eprintln!("segment {i}: {got} vs {}", 1.0 / w);
        assert!(
            (got * w - 1.0).abs() < 0.02,
            "segment {i}: {got} vs {}",
            1.0 / w
        );
    }
}

// Bar: a 2000-point Swiss roll's fuzzy kNN graph is unrolled (rank correlation of
// sheet vs layout distances) >= 0.9.
#[test]
fn a_swiss_roll_graph_is_unrolled() {
    let (data, sheet) = swiss_roll(2000, 4);
    let edges = fuzzy_edges(&data, 15);
    let y = stress_layout_2d_on_graph(2000, &edges, 4);
    let rho = unrolled(&sheet, &flat(&y));
    eprintln!("swiss roll unrolled {rho}");
    assert!(rho >= 0.9, "swiss roll unrolled {rho}");
}

// Bar: two lattices, a chain longer than either and two lone nodes; no
// point falls in another piece's bounding box, no two lone nodes coincide,
// and each lattice is still laid out as its grid (rank correlation >= 0.98).
#[test]
fn separate_pieces_do_not_overlap() {
    let (mut edges, place_a) = lattice(8, 5, 0);
    let (edges_b, place_b) = lattice(6, 6, 40);
    edges.extend(edges_b);
    edges.extend((76..105).map(|i| (i, i + 1, 1.0)));
    let n = 40 + 36 + 30 + 2;
    let y = stress_layout_2d_on_graph(n, &edges, 7);
    assert!(y.iter().all(|v| v.is_finite()));
    let pieces: Vec<Vec<usize>> = vec![
        (0..40).collect(),
        (40..76).collect(),
        (76..106).collect(),
        vec![106],
        vec![107],
    ];
    for (p, piece) in pieces.iter().enumerate() {
        let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
        for &i in piece {
            for c in 0..2 {
                lo[c] = lo[c].min(y[(i, c)]);
                hi[c] = hi[c].max(y[(i, c)]);
            }
        }
        for (q, other) in pieces.iter().enumerate() {
            if q == p {
                continue;
            }
            for &i in other {
                let inside = (0..2).all(|c| y[(i, c)] >= lo[c] && y[(i, c)] <= hi[c]);
                assert!(!inside, "node {i} of piece {q} falls in piece {p}");
            }
        }
    }
    let a = faithful(&y, &pieces[0], &place_a);
    let b = faithful(&y, &pieces[1], &place_b);
    eprintln!("pieces: lattice a {a}, lattice b {b}");
    assert!(a >= 0.98 && b >= 0.98, "lattices {a} {b}");
}

// Bar: shapes are n x 2 and finite for n = 0, 1, 2 with and without an
// edge, and for edgeless n = 3; two joined nodes sit 1/w apart (within 1%);
// no two nodes coincide.
#[test]
fn tiny_inputs() {
    assert_eq!(stress_layout_2d_on_graph(0, &[], 1).shape(), (0, 2));
    let one = stress_layout_2d_on_graph(1, &[], 1);
    assert_eq!(one.shape(), (1, 2));
    assert!(one.iter().all(|v| v.is_finite()));
    let joined = stress_layout_2d_on_graph(2, &[(0, 1, 0.5)], 1);
    assert_eq!(joined.shape(), (2, 2));
    eprintln!("two joined nodes {} apart", dist(&joined, 0, 1));
    assert!((dist(&joined, 0, 1) - 2.0).abs() < 0.02);
    for (n, edges) in [(2, vec![]), (3, vec![])] {
        let y = stress_layout_2d_on_graph(n, &edges, 1);
        assert_eq!(y.shape(), (n, 2));
        assert!(y.iter().all(|v| v.is_finite()));
        for i in 0..n {
            for j in 0..i {
                assert!(dist(&y, i, j) > 0.0, "nodes {i} and {j} coincide");
            }
        }
    }
}

// Bar: the same seed gives a bit-identical layout.
#[test]
fn same_seed_same_layout() {
    let (data, _) = swiss_roll(300, 9);
    let edges = fuzzy_edges(&data, 10);
    let a = stress_layout_2d_on_graph(300, &edges, 42);
    let b = stress_layout_2d_on_graph(300, &edges, 42);
    assert_eq!(a, b);
}

// Bar: multiplying every weight by 100 shrinks the layout (pieces and gaps
// alike) by exactly 100, within 1e-3 of its extent.
#[test]
fn the_layout_scales_with_the_weights() {
    let (mut edges, _) = lattice(5, 4, 0);
    edges.push((20, 21, 1.0));
    let n = 23;
    let a = stress_layout_2d_on_graph(n, &edges, 2);
    let scaled: Vec<_> = edges.iter().map(|&(i, j, w)| (i, j, 100.0 * w)).collect();
    let b = stress_layout_2d_on_graph(n, &scaled, 2) * 100.0;
    let extent = a.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
    let worst = (a - b).iter().fold(0.0_f32, |m, v| m.max(v.abs()));
    eprintln!("scale mismatch {worst} of extent {extent}");
    assert!(
        worst <= 1e-3 * extent,
        "scale mismatch {worst} of extent {extent}"
    );
}

// Bar: an edge of weight zero is no edge: the layout is finite and the
// same as without it.
#[test]
fn a_zero_weight_is_no_edge() {
    let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 0.0)];
    let with = stress_layout_2d_on_graph(4, &edges, 6);
    let without = stress_layout_2d_on_graph(4, &edges[..2], 6);
    assert!(with.iter().all(|v| v.is_finite()));
    assert_eq!(with, without);
}

// Bar: the leaves of a star (all the same distance from each other, so
// alike to the MDS start) end apart and around the hub: no two nodes
// closer than 0.3, and the spokes equal within 5% (six leaves 2 apart do
// not fit at radius 1, so stress stretches every spoke alike).
#[test]
fn a_star_spreads_its_leaves() {
    let edges: Vec<_> = (1..7).map(|i| (0, i, 1.0)).collect();
    let y = stress_layout_2d_on_graph(7, &edges, 8);
    let mut closest = f32::MAX;
    for i in 0..7 {
        for j in 0..i {
            closest = closest.min(dist(&y, i, j));
        }
    }
    let spokes: Vec<f32> = (1..7).map(|i| dist(&y, 0, i)).collect();
    let (lo, hi) = spokes
        .iter()
        .fold((f32::MAX, 0.0_f32), |(lo, hi), &s| (lo.min(s), hi.max(s)));
    let spread = hi / lo - 1.0;
    eprintln!("star: closest pair {closest}, spokes {lo}..{hi}, spread {spread}");
    assert!(
        closest >= 0.3 && spread <= 0.05,
        "star: closest {closest}, spread {spread}"
    );
}

/// Mean squared relative error of the layout's distances against the
/// shortest-path lengths (edge length `1/w`, repeated edges summed first),
/// over all pairs.
fn relative_stress(n: usize, edges: &[(usize, usize, f32)], y: &Mat) -> f64 {
    let d = shortest_paths(&PreparedGraph::new(n, edges));
    let mut s = 0.0;
    for i in 0..n {
        for j in 0..i {
            let dij = f64::from(d[i * n + j]);
            s += ((f64::from(dist(y, i, j)) - dij) / dij).powi(2);
        }
    }
    s / (n * (n - 1) / 2) as f64
}

// Bar: on a random tree-plus-chords graph of 500 nodes, relative stress
// <= 0.14 (visiting pairs in one fixed order reaches 0.144).
#[test]
fn random_pair_order_lowers_stress() {
    let mut rng = SmallRng::seed_from_u64(3);
    let mut edges = vec![];
    for i in 1..500 {
        for _ in 0..2 {
            edges.push((rng.random_range(0..i), i, rng.random_range(0.2_f32..1.0)));
        }
    }
    let s = relative_stress(500, &edges, &stress_layout_2d_on_graph(500, &edges, 0));
    eprintln!("random graph relative stress {s}");
    assert!(s <= 0.14, "random graph relative stress {s}");
}

// Bar: twins (two nodes with the same neighbours at the same lengths) have
// the same shortest-path rows, so the MDS start puts them on one point; the
// layout stays finite.
#[test]
fn twins_that_start_on_one_point_stay_finite() {
    let edges = vec![
        (0, 2, 1.0),
        (0, 3, 1.0),
        (1, 2, 1.0),
        (1, 3, 1.0),
        (2, 3, 1.0),
    ];
    let y = stress_layout_2d_on_graph(4, &edges, 1);
    assert!(y.iter().all(|v| v.is_finite()), "{y}");
}
