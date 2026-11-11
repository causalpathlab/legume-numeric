use super::*;

/// A stem of `stem` nodes from node `start`, then two branches of `arm`
/// nodes each; which branch takes the lower id alternates by level, so an
/// order by id zigzags between them. Returns the edges and each node's branch (0 stem, 1 or 2).
fn fork(stem: usize, arm: usize) -> (usize, Vec<(usize, usize, f32)>, Vec<u8>) {
    let n = stem + 2 * arm;
    let mut branch = vec![0_u8; n];
    let mut edges: Vec<(usize, usize, f32)> = (1..stem).map(|i| (i - 1, i, 1.0)).collect();
    for k in 0..arm {
        // Which branch takes the lower id alternates by level.
        let (lo, hi) = (stem + 2 * k, stem + 2 * k + 1);
        let (a, b) = if k % 2 == 0 { (lo, hi) } else { (hi, lo) };
        branch[a] = 1;
        branch[b] = 2;
        let (pa, pb) = if k == 0 {
            (stem - 1, stem - 1)
        } else {
            let (plo, phi) = (lo - 2, hi - 2);
            if k % 2 == 1 {
                (plo, phi)
            } else {
                (phi, plo)
            }
        };
        edges.push((pa.min(a), pa.max(a), 1.0));
        edges.push((pb.min(b), pb.max(b), 1.0));
    }
    (n, edges, branch)
}

#[test]
fn a_fork_is_laid_out_from_its_stem_down_with_branches_apart() {
    // The stem is longer than the branches, so the longest path runs from
    // its start, where the automatic root goes. (With short stems it runs
    // tip to tip, and a root has to be given.)
    let (n, edges, branch) = fork(20, 8);
    let y = sugiyama_layout_2d_on_graph(n, &edges, None);
    assert_eq!(y.shape(), (n, 2));
    let layer = |i: usize| y[(i, 1)];
    for i in 0..20 {
        assert_eq!(layer(i), -(i as f32), "stem {i}");
    }
    // In every layer the two branches share, they sit on opposite sides.
    let side = |b: u8| -> Vec<(f32, f32)> {
        (0..n)
            .filter(|&i| branch[i] == b)
            .map(|i| (layer(i), y[(i, 0)]))
            .collect()
    };
    let (one, two) = (side(1), side(2));
    let sign = |l: f32| {
        let a = one.iter().find(|p| p.0 == l).map(|p| p.1);
        let b = two.iter().find(|p| p.0 == l).map(|p| p.1);
        a.zip(b).map(|(a, b)| (a - b).signum())
    };
    let signs: Vec<f32> = one.iter().filter_map(|p| sign(p.0)).collect();
    assert!(!signs.is_empty());
    assert!(
        signs.iter().all(|&s| s == signs[0] && s != 0.0),
        "{signs:?}"
    );
}

#[test]
fn every_edge_spans_at_most_one_layer_and_layers_follow_the_root() {
    let (n, edges, _) = fork(6, 8);
    let y = sugiyama_layout_2d_on_graph(n, &edges, Some(0));
    assert_eq!(y[(0, 1)], 0.0, "the root is the top layer");
    for &(i, j, _) in &edges {
        assert!((y[(i, 1)] - y[(j, 1)]).abs() <= 1.0, "edge {i}-{j}");
    }
    // Down the stem from the given root, a layer a step.
    for i in 0..6 {
        assert_eq!(y[(i, 1)], -(i as f32), "node {i}");
    }
}

#[test]
fn nodes_of_one_layer_do_not_share_a_place() {
    let (n, edges, _) = fork(4, 6);
    let y = sugiyama_layout_2d_on_graph(n, &edges, Some(0));
    for i in 0..n {
        for j in 0..i {
            assert!((y.row(i) - y.row(j)).norm() > 0.0, "{i} and {j}");
        }
    }
}

#[test]
fn separate_pieces_sit_side_by_side() {
    // Two chains with no edge between them.
    let mut edges: Vec<(usize, usize, f32)> = (1..5).map(|i| (i - 1, i, 1.0)).collect();
    edges.extend((6..10).map(|i| (i - 1, i, 1.0)));
    let y = sugiyama_layout_2d_on_graph(10, &edges, None);
    let span = |r: std::ops::Range<usize>| {
        let xs: Vec<f32> = r.map(|i| y[(i, 0)]).collect();
        (
            xs.iter().cloned().fold(f32::MAX, f32::min),
            xs.iter().cloned().fold(f32::MIN, f32::max),
        )
    };
    let (a, b) = (span(0..5), span(5..10));
    assert!(a.1 < b.0 || b.1 < a.0, "{a:?} and {b:?} overlap");
}

#[test]
fn tiny_inputs_get_a_layout_of_their_size() {
    for n in [0, 1, 2, 5] {
        let edges: Vec<(usize, usize, f32)> = (1..n).map(|i| (i - 1, i, 1.0)).collect();
        let y = sugiyama_layout_2d_on_graph(n, &edges, None);
        assert_eq!(y.shape(), (n, 2));
        assert!(y.iter().all(|v| v.is_finite()));
    }
    // A node with no edge at all.
    let y = sugiyama_layout_2d_on_graph(3, &[(0, 1, 1.0)], None);
    assert!(y.iter().all(|v| v.is_finite()));
}
