use super::super::*;
use super::*;

/// A `w x h` grid with its diagonals (so every inner node has at least
/// three neighbours), weights that vary from edge to edge, and features
/// that place each node on a gently curved sheet.
fn grid(w: usize, h: usize) -> (usize, Vec<(usize, usize, f32)>, Mat) {
    let id = |x: usize, y: usize| y * w + x;
    let mut edges = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let weight = |k: usize| 1.0 + 0.1 * ((id(x, y) * 7 + k) % 5) as f32;
            if x + 1 < w {
                edges.push((id(x, y), id(x + 1, y), weight(0)));
            }
            if y + 1 < h {
                edges.push((id(x, y), id(x, y + 1), weight(1)));
            }
            if x + 1 < w && y + 1 < h {
                edges.push((id(x, y), id(x + 1, y + 1), weight(2)));
            }
        }
    }
    let n = w * h;
    let features = Mat::from_fn(n, 3, |i, d| {
        let (x, y) = ((i % w) as f32, (i / w) as f32);
        [x, y, 0.3 * x.sin()][d]
    });
    (n, edges, features)
}

type Layout = Box<dyn Fn(&[(usize, usize, f32)]) -> Mat>;

// Bar: in every graph layout, an edge of weight zero, below zero, NaN or
// infinite is no edge: the layout is the same as without it.
#[test]
fn a_weight_not_above_zero_or_not_finite_is_no_edge_in_any_layout() {
    let (n, kept, features) = grid(5, 4);
    let mut noisy = kept.clone();
    noisy.extend([
        (0, 19, 0.0),
        (3, 10, -1.0),
        (2, 15, f32::NAN),
        (4, 12, f32::INFINITY),
    ]);
    let layouts: Vec<(&str, Layout)> = vec![
        (
            "stress",
            Box::new(move |e| stress_layout_2d_on_graph(n, e, 3)),
        ),
        (
            "forceatlas2",
            Box::new(move |e| forceatlas2_layout_2d_on_graph(n, e, 50, 3)),
        ),
        (
            "diffusion",
            Box::new(move |e| diffusion_map_2d_on_graph(n, e, 1)),
        ),
        (
            "radial",
            Box::new(move |e| radial_tree_layout_2d_on_graph(n, e, None)),
        ),
        (
            "sugiyama",
            Box::new(move |e| sugiyama_layout_2d_on_graph(n, e, None)),
        ),
        (
            "lle",
            Box::new(move |e| lle_layout_2d_on_graph(&features, e)),
        ),
    ];
    for (name, layout) in &layouts {
        let (with, without) = (layout(&noisy), layout(&kept));
        assert!(without.iter().all(|v| v.is_finite()), "{name}: {without}");
        assert_eq!(with, without, "{name}");
    }
}

// Bar: preparing drops self-loops and edges of weight not above zero or not
// finite, orders each edge (low, high), sums repeats, and lists each node's
// neighbours sorted.
#[test]
fn preparing_cleans_and_sorts_the_edges() {
    let edges = [
        (2, 0, 1.0),
        (3, 2, 2.0),
        (0, 2, 0.5),
        (1, 1, 3.0),
        (1, 3, 0.0),
        (3, 1, -2.0),
        (0, 3, f32::NAN),
        (2, 3, f32::INFINITY),
        (0, 4, 1.0),
    ];
    let g = PreparedGraph::new(5, &edges);
    assert_eq!(g.n(), 5);
    assert_eq!(g.edges(), &[(0, 2, 1.5), (0, 4, 1.0), (2, 3, 2.0)]);
    assert_eq!(g.neighbours(0), &[(2, 1.5), (4, 1.0)]);
    assert_eq!(g.neighbours(1), &[]);
    assert_eq!(g.neighbours(2), &[(0, 1.5), (3, 2.0)]);
    assert_eq!(g.neighbour_ids(3).collect::<Vec<_>>(), vec![2]);
}

#[test]
#[should_panic(expected = "not below the node count")]
fn an_edge_end_out_of_range_panics() {
    let _ = PreparedGraph::new(3, &[(0, 3, 1.0)]);
}

/// The gap between two boxes `[x_lo, y_lo, x_hi, y_hi]`: the larger of
/// their separations along x and y, negative when they overlap.
fn box_gap(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let gx = (b[0] - a[2]).max(a[0] - b[2]);
    let gy = (b[1] - a[3]).max(a[1] - b[3]);
    gx.max(gy)
}

/// Each piece laid out as a slanted line, node `k` at `(2k, k)`; a piece of
/// one node must never reach it.
fn slant(g: &PreparedGraph, _: Option<usize>) -> Mat {
    assert!(g.n() > 1, "a lone node was laid out");
    Mat::from_fn(g.n(), 2, |k, d| [2.0 * k as f32, k as f32][d])
}

// Bar: pieces are placed with a gap of at least a tenth of the largest
// box's longer side between every two boxes; lone nodes are not laid out
// by the piece layout; without `first` the packing's lower corner is the
// origin, and with it that piece keeps its own coordinates and learns its
// local number.
#[test]
fn pieces_are_placed_apart_and_the_first_keeps_its_place() {
    let edges = [(0, 1, 1.0), (2, 3, 1.0), (3, 4, 1.0), (6, 7, 1.0)];
    let g = PreparedGraph::new(8, &edges);
    let pieces: [&[usize]; 4] = [&[0, 1], &[2, 3, 4], &[5], &[6, 7]];
    let largest = 4.0_f32;

    let y = lay_out_by_piece(&g, slant, None);
    let boxes: Vec<[f32; 4]> = pieces
        .iter()
        .map(|p| bounding_box(&Mat::from_fn(p.len(), 2, |k, d| y[(p[k], d)])))
        .collect();
    for a in 0..boxes.len() {
        for b in (a + 1)..boxes.len() {
            let gap = box_gap(&boxes[a], &boxes[b]);
            assert!(gap >= 0.1 * largest - 1e-6, "pieces {a} and {b}: {gap}");
        }
    }
    let all = bounding_box(&y);
    assert_eq!((all[0], all[1]), (0.0, 0.0));

    let seen = std::sync::Mutex::new(Vec::new());
    let y = lay_out_by_piece(
        &g,
        |p, at| {
            seen.lock().unwrap().push((p.n(), at));
            slant(p, at)
        },
        Some(3),
    );
    for (k, &i) in pieces[1].iter().enumerate() {
        assert_eq!((y[(i, 0)], y[(i, 1)]), (2.0 * k as f32, k as f32));
    }
    let mut seen = seen.into_inner().unwrap();
    seen.sort_unstable();
    assert_eq!(seen, vec![(2, None), (2, None), (3, Some(1))]);
}

// Bar: when every piece is a lone node the gap is 1, so lone nodes sit
// at least 1 apart.
#[test]
fn lone_nodes_sit_one_apart() {
    let y = lay_out_by_piece(&PreparedGraph::new(5, &[]), slant, None);
    for i in 0..5 {
        for j in 0..i {
            assert!((y.row(i) - y.row(j)).norm() >= 1.0 - 1e-6, "{i} {j}");
        }
    }
}
