use super::*;
use crate::matrix::landmark_init::tests::swiss_roll;
use crate::matrix::test_support::fuzzy_edges;
use std::f32::consts::PI;

fn radius(y: &Mat, i: usize) -> f32 {
    y[(i, 0)].hypot(y[(i, 1)])
}

fn angle(y: &Mat, i: usize) -> f32 {
    y[(i, 1)].atan2(y[(i, 0)])
}

/// The smaller angle between two directions, in `[0, π]`.
fn angle_between(a: f32, b: f32) -> f32 {
    let d = (a - b).rem_euclid(2.0 * PI);
    d.min(2.0 * PI - d)
}

fn dist(y: &Mat, i: usize, j: usize) -> f32 {
    (y[(i, 0)] - y[(j, 0)]).hypot(y[(i, 1)] - y[(j, 1)])
}

fn all_finite(y: &Mat) -> bool {
    y.iter().all(|v| v.is_finite())
}

fn min_pair_distance(y: &Mat) -> f32 {
    let n = y.nrows();
    let mut best = f32::INFINITY;
    for i in 0..n {
        for j in (i + 1)..n {
            best = best.min(dist(y, i, j));
        }
    }
    best
}

/// Edges of a path through `nodes` in the given order, weight `w`.
fn path(nodes: &[usize], w: f32) -> Vec<(usize, usize, f32)> {
    nodes.windows(2).map(|p| (p[0], p[1], w)).collect()
}

/// Bar: each of three arms of five nodes lies along one direction (every
/// node within 1e-4 rad of the arm's first node), the arms point at least
/// π/2 apart (2π/3 expected), the centre is at the origin, and the k-th node
/// out sits at radius k (within 1e-4).
#[test]
fn a_star_of_three_long_arms_is_three_spokes() {
    let arm_len = 5;
    let arms: Vec<Vec<usize>> = (0..3)
        .map(|a| (0..arm_len).map(|k| 1 + a * arm_len + k).collect())
        .collect();
    let mut edges = vec![];
    for arm in &arms {
        let mut nodes = vec![0];
        nodes.extend(arm);
        edges.extend(path(&nodes, 1.0));
    }
    let y = radial_tree_layout_2d_on_graph(1 + 3 * arm_len, &edges, None);

    assert!(radius(&y, 0) < 1e-5, "centre at {}", radius(&y, 0));
    let mut directions = vec![];
    for arm in &arms {
        let dir = angle(&y, arm[0]);
        for (k, &v) in arm.iter().enumerate() {
            let off = angle_between(angle(&y, v), dir);
            assert!(off < 1e-4, "node {v} is {off} rad off its arm");
            let r = radius(&y, v);
            assert!((r - (k + 1) as f32).abs() < 1e-4, "node {v} at radius {r}");
        }
        directions.push(dir);
    }
    for a in 0..3 {
        for b in (a + 1)..3 {
            let gap = angle_between(directions[a], directions[b]);
            assert!(gap > PI / 2.0, "arms {a} and {b} only {gap} rad apart");
        }
    }
}

/// Bar: two chains of five (weight 0.5) joined by a strong (0.9) and a weak
/// (0.1) cross edge. The tree keeps the strong one, so its ends sit one unit
/// apart (within 1e-4), and drops the weak one, so its ends sit more than 5
/// apart (9 expected: the far ends of a ten-node path).
#[test]
fn the_tree_keeps_the_strong_edges() {
    let mut edges = path(&[0, 1, 2, 3, 4], 0.5);
    edges.extend(path(&[5, 6, 7, 8, 9], 0.5));
    edges.push((0, 5, 0.1));
    edges.push((4, 9, 0.9));
    let y = radial_tree_layout_2d_on_graph(10, &edges, None);
    let strong = dist(&y, 4, 9);
    let weak = dist(&y, 0, 5);
    assert!((strong - 1.0).abs() < 1e-4, "strong edge spans {strong}");
    assert!(weak > 5.0, "weak edge spans only {weak}");
}

/// Bar: on a path of seven nodes, labelled out of order, the middle node is
/// at the origin (within 1e-5) and both ends at radius 3 (within 1e-4).
#[test]
fn the_automatic_root_is_the_centre_of_a_path() {
    let order = [5, 2, 6, 0, 3, 1, 4];
    let y = radial_tree_layout_2d_on_graph(7, &path(&order, 1.0), None);
    assert!(radius(&y, 0) < 1e-5, "middle at {}", radius(&y, 0));
    for end in [5, 4] {
        let r = radius(&y, end);
        assert!((r - 3.0).abs() < 1e-4, "end {end} at radius {r}");
    }
}

/// Bar: a given root sits at the origin (within 1e-5), also when the graph
/// has a second piece, and the other end of its path is at radius 6.
#[test]
fn a_given_root_sits_at_the_origin() {
    let order = [5, 2, 6, 0, 3, 1, 4];
    let mut edges = path(&order, 1.0);
    edges.extend(path(&[7, 8, 9], 1.0));
    for root in [5, 3] {
        let y = radial_tree_layout_2d_on_graph(10, &edges, Some(root));
        assert!(
            radius(&y, root) < 1e-5,
            "root {root} at {}",
            radius(&y, root)
        );
    }
    let y = radial_tree_layout_2d_on_graph(10, &edges, Some(5));
    let r = radius(&y, 4);
    assert!((r - 6.0).abs() < 1e-4, "far end at radius {r}");
}

/// Bar: a subtree's wedge follows its number of leaves. The root has a child
/// `a` with three leaves and a leaf child `b`, so `a` gets 3/4 of the circle
/// and `b` 1/4: the four leaves (`b` and those of `a`) are evenly spread,
/// each π/2 from the next (within 1e-4; an equal split per child would put
/// the leaves of `a` π/3 apart), and `a` sits at the angle of its middle
/// leaf and π/2 from the other two (the middle of its wedge).
#[test]
fn wedges_follow_leaf_counts() {
    // root 0, a = 1, b = 2, leaves of a = 3, 4, 5.
    let edges = vec![
        (0, 1, 1.0),
        (0, 2, 1.0),
        (1, 3, 1.0),
        (1, 4, 1.0),
        (1, 5, 1.0),
    ];
    let y = radial_tree_layout_2d_on_graph(6, &edges, Some(0));
    let mut leaves: Vec<f32> = [3, 4, 5, 2].iter().map(|&v| angle(&y, v)).collect();
    leaves.sort_by(f32::total_cmp);
    for k in 0..4 {
        let gap = angle_between(leaves[k], leaves[(k + 1) % 4]);
        assert!((gap - PI / 2.0).abs() < 1e-4, "leaves {gap} rad apart");
    }
    // In the middle of its wedge, `a` is on its middle leaf and π/2 from
    // the other two; at the start of its wedge it would be π from one.
    let to_leaves: Vec<f32> = [3, 4, 5]
        .iter()
        .map(|&v| angle_between(angle(&y, 1), angle(&y, v)))
        .collect();
    let far = to_leaves.iter().copied().fold(0.0, f32::max);
    let near = to_leaves.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(near < 1e-4, "a is {near} rad off its middle leaf");
    assert!((far - PI / 2.0).abs() < 1e-4, "a is {far} rad from a leaf");
}

/// Bar: three pieces (a star, a path and a lone node) get bounding boxes
/// that do not overlap; each pair is apart along x or y by at least a
/// tenth of the largest box's longer side (here 3, the path's length).
#[test]
fn separate_pieces_do_not_overlap() {
    let mut edges = vec![(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (0, 4, 1.0)];
    edges.extend(path(&[5, 6, 7, 8], 1.0));
    let pieces: [&[usize]; 3] = [&[0, 1, 2, 3, 4], &[5, 6, 7, 8], &[9]];
    let y = radial_tree_layout_2d_on_graph(10, &edges, None);
    assert!(all_finite(&y));
    let bbox = |p: &[usize]| {
        let col = |c: usize| {
            let v: Vec<f32> = p.iter().map(|&i| y[(i, c)]).collect();
            let lo = v.iter().copied().fold(f32::INFINITY, f32::min);
            let hi = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            (lo, hi)
        };
        (col(0), col(1))
    };
    for a in 0..3 {
        for b in (a + 1)..3 {
            let ((ax, ay), (bx, by)) = (bbox(pieces[a]), bbox(pieces[b]));
            let gap_x = (bx.0 - ax.1).max(ax.0 - bx.1);
            let gap_y = (by.0 - ay.1).max(ay.0 - by.1);
            let gap = gap_x.max(gap_y);
            assert!(gap >= 0.3 - 1e-5, "pieces {a} and {b} only {gap} apart");
        }
    }
}

/// Bar: n = 0 gives a 0 x 2 matrix; n = 1 and n = 2 (joined or not) and a
/// node with no edges give finite places, all distinct (two lone nodes at
/// least 1 apart; beside a pair one unit long, at least the gap of a tenth
/// of it), and a joined pair is one unit apart.
#[test]
fn tiny_inputs_and_lone_nodes_are_finite() {
    let y = radial_tree_layout_2d_on_graph(0, &[], None);
    assert_eq!(y.shape(), (0, 2));

    let y = radial_tree_layout_2d_on_graph(1, &[], None);
    assert_eq!(y.shape(), (1, 2));
    assert!(all_finite(&y));

    let y = radial_tree_layout_2d_on_graph(2, &[(0, 1, 0.3)], None);
    assert!(all_finite(&y));
    assert!((dist(&y, 0, 1) - 1.0).abs() < 1e-4);

    let y = radial_tree_layout_2d_on_graph(2, &[], None);
    assert!(all_finite(&y));
    assert!(dist(&y, 0, 1) >= 1.0 - 1e-5, "{}", dist(&y, 0, 1));

    // Node 2 has no edges; node 3 only a self loop.
    let y = radial_tree_layout_2d_on_graph(4, &[(0, 1, 1.0), (3, 3, 1.0)], Some(2));
    assert!(all_finite(&y));
    assert!(
        min_pair_distance(&y) >= 0.1 - 1e-5,
        "{}",
        min_pair_distance(&y)
    );
    assert!(radius(&y, 2) < 1e-5);
}

/// Bar: on the fuzzy kNN graph of a 400-point Swiss roll, every place is
/// finite and no two nodes share one (closest pair more than 1e-4 apart),
/// and each node is at radius equal to its tree depth: tree neighbours'
/// radii differ by one, checked as every non-root radius being a whole
/// number (within 1e-3).
#[test]
fn no_two_nodes_share_a_place() {
    let (data, _) = swiss_roll(400, 3);
    let edges = fuzzy_edges(&data, 15);
    let y = radial_tree_layout_2d_on_graph(400, &edges, None);
    assert!(all_finite(&y));
    let closest = min_pair_distance(&y);
    assert!(closest > 1e-4, "closest pair {closest} apart");
    let roots = (0..400).filter(|&i| radius(&y, i) < 1e-5).count();
    assert_eq!(roots, 1, "the roll is one piece with one root");
    for i in 0..400 {
        let r = radius(&y, i);
        assert!((r - r.round()).abs() < 1e-3, "node {i} at radius {r}");
    }
}

#[test]
#[should_panic]
fn a_root_out_of_range_panics() {
    let _ = radial_tree_layout_2d_on_graph(3, &[(0, 1, 1.0)], Some(3));
}

/// Bar: a hub with 200,000 spokes, its edges listed hub first, is laid out
/// in under 2 s with the hub at the origin. Without path halving in the
/// union-find each find walks a chain one longer than the last, and this
/// takes quadratic time.
#[test]
fn a_big_hub_is_laid_out_fast() {
    let n = 200_001;
    let edges: Vec<(usize, usize, f32)> = (1..n).map(|i| (0, i, 1.0)).collect();
    let t = std::time::Instant::now();
    let y = radial_tree_layout_2d_on_graph(n, &edges, None);
    let secs = t.elapsed().as_secs_f32();
    assert!(secs < 2.0, "took {secs} s");
    assert!(radius(&y, 0) < 1e-5);
}

// Bar: an edge of weight zero or less is no edge, as in the other graph
// layouts: the same layout as without it.
#[test]
fn a_non_positive_weight_is_no_edge() {
    let edges = vec![
        (0, 1, 1.0),
        (1, 2, 1.0),
        (2, 3, 0.0),
        (3, 4, -1.0),
        (4, 5, 1.0),
    ];
    let kept: Vec<_> = edges.iter().copied().filter(|e| e.2 > 0.0).collect();
    assert_eq!(
        radial_tree_layout_2d_on_graph(6, &edges, None),
        radial_tree_layout_2d_on_graph(6, &kept, None)
    );
}
