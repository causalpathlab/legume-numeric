use super::*;
use rand::{rngs::SmallRng, RngExt, SeedableRng};

const ITERS: usize = 1000;

/// `k` groups of `size` nodes: each node linked with weight 1 to `per` random
/// others of its group, plus `bridges` weak (0.05) edges between consecutive
/// groups.
fn groups(
    k: usize,
    size: usize,
    per: usize,
    bridges: usize,
    seed: u64,
) -> Vec<(usize, usize, f32)> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut edges = vec![];
    for g in 0..k {
        for a in 0..size {
            for _ in 0..per {
                let b = rng.random_range(0..size);
                if a != b {
                    edges.push((g * size + a, g * size + b, 1.0));
                }
            }
        }
        for _ in 0..bridges {
            let h = (g + 1) % k;
            let a = g * size + rng.random_range(0..size);
            let b = h * size + rng.random_range(0..size);
            edges.push((a, b, 0.05));
        }
    }
    edges
}

fn chain(n: usize) -> Vec<(usize, usize, f32)> {
    (1..n).map(|i| (i - 1, i, 1.0)).collect()
}

fn dist(y: &Mat, i: usize, j: usize) -> f32 {
    ((y[(i, 0)] - y[(j, 0)]).powi(2) + (y[(i, 1)] - y[(j, 1)]).powi(2)).sqrt()
}

fn centroid(y: &Mat, nodes: &[usize]) -> [f32; 2] {
    let m = nodes.len() as f32;
    [
        nodes.iter().map(|&i| y[(i, 0)]).sum::<f32>() / m,
        nodes.iter().map(|&i| y[(i, 1)]).sum::<f32>() / m,
    ]
}

/// Bounding box `[xmin, xmax, ymin, ymax]` of `nodes`.
fn bbox(y: &Mat, nodes: &[usize]) -> [f32; 4] {
    let mut b = [
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ];
    for &i in nodes {
        b[0] = b[0].min(y[(i, 0)]);
        b[1] = b[1].max(y[(i, 0)]);
        b[2] = b[2].min(y[(i, 1)]);
        b[3] = b[3].max(y[(i, 1)]);
    }
    b
}

/// Gap between two boxes `[xmin, xmax, ymin, ymax]`: the larger of their
/// separations along x and y, negative when they overlap.
fn box_gap(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    (b[0] - a[1])
        .max(a[0] - b[1])
        .max(b[2] - a[3])
        .max(a[2] - b[3])
}

fn ranks(v: &[f32]) -> Vec<f32> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[a].total_cmp(&v[b]));
    let mut r = vec![0.0; v.len()];
    let mut s = 0;
    while s < idx.len() {
        let mut e = s;
        while e + 1 < idx.len() && v[idx[e + 1]] == v[idx[s]] {
            e += 1;
        }
        let mid = (s + e) as f32 / 2.0;
        for &k in &idx[s..=e] {
            r[k] = mid;
        }
        s = e + 1;
    }
    r
}

fn spearman(a: &[f32], b: &[f32]) -> f32 {
    let (ra, rb) = (ranks(a), ranks(b));
    let n = a.len() as f32;
    let (ma, mb) = (ra.iter().sum::<f32>() / n, rb.iter().sum::<f32>() / n);
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (x, y) in ra.iter().zip(&rb) {
        sab += (x - ma) * (y - mb);
        saa += (x - ma).powi(2);
        sbb += (y - mb).powi(2);
    }
    sab / (saa * sbb).sqrt()
}

fn all_finite(y: &Mat) -> bool {
    y.iter().all(|v| v.is_finite())
}

/// Bar: the distance between any two group centres is at least 3 times the
/// mean distance of a node to its own group's centre.
#[test]
fn three_groups_stay_apart() {
    let size = 25;
    let edges = groups(3, size, 4, 3, 1);
    let y = forceatlas2_layout_2d_on_graph(3 * size, &edges, ITERS, 7);
    assert_eq!(y.shape(), (3 * size, 2));
    assert!(all_finite(&y));
    let members: Vec<Vec<usize>> = (0..3)
        .map(|g| (g * size..(g + 1) * size).collect())
        .collect();
    let cents: Vec<[f32; 2]> = members.iter().map(|m| centroid(&y, m)).collect();
    let mut spread = 0.0;
    for (m, c) in members.iter().zip(&cents) {
        for &i in m {
            spread += ((y[(i, 0)] - c[0]).powi(2) + (y[(i, 1)] - c[1]).powi(2)).sqrt();
        }
    }
    spread /= (3 * size) as f32;
    let mut min_gap = f32::INFINITY;
    for a in 0..3 {
        for b in (a + 1)..3 {
            let d =
                ((cents[a][0] - cents[b][0]).powi(2) + (cents[a][1] - cents[b][1]).powi(2)).sqrt();
            min_gap = min_gap.min(d);
        }
    }
    eprintln!(
        "three groups: min centre gap / spread = {}",
        min_gap / spread
    );
    assert!(spread > 0.0);
    assert!(min_gap >= 3.0 * spread, "gap {min_gap} spread {spread}");
}

/// Bar: Spearman correlation between chain distance |i - j| and layout
/// distance over all pairs is at least 0.9.
#[test]
fn chain_keeps_its_order() {
    let n = 40;
    let y = forceatlas2_layout_2d_on_graph(n, &chain(n), ITERS, 3);
    let (mut a, mut b) = (vec![], vec![]);
    for i in 0..n {
        for j in (i + 1)..n {
            a.push((j - i) as f32);
            b.push(dist(&y, i, j));
        }
    }
    let rho = spearman(&a, &b);
    eprintln!("chain: spearman = {rho}");
    assert!(rho >= 0.9, "rho {rho}");
}

/// Bar: the bounding boxes of two pieces with no edge between them are
/// separated by a gap of at least 5% of the larger box's longer side, and
/// each piece is still laid out (its nodes are not collapsed).
#[test]
fn separate_pieces_do_not_overlap() {
    let size = 20;
    let mut edges = groups(1, size, 4, 0, 2);
    edges.extend(
        chain(size)
            .into_iter()
            .map(|(i, j, w)| (i + size, j + size, w)),
    );
    let y = forceatlas2_layout_2d_on_graph(2 * size, &edges, ITERS, 5);
    assert!(all_finite(&y));
    let p: Vec<usize> = (0..size).collect();
    let q: Vec<usize> = (size..2 * size).collect();
    let (a, b) = (bbox(&y, &p), bbox(&y, &q));
    let extent = [a[1] - a[0], a[3] - a[2], b[1] - b[0], b[3] - b[2]]
        .into_iter()
        .fold(0.0, f32::max);
    let gap = box_gap(&a, &b);
    eprintln!("pieces: gap / extent = {}", gap / extent);
    assert!(gap >= 0.05 * extent, "gap {gap} extent {extent}");
    assert!(a[1] - a[0] > 0.0 && b[1] - b[0] > 0.0);
}

/// Bar: many pieces (one group and 30 lone nodes) pack into a box whose
/// longer side is at most 3 times the shorter one, and no two pieces'
/// bounding boxes touch. A graph of 25 nodes and no edge packs to the same
/// aspect bound.
#[test]
fn many_pieces_pack_compactly() {
    let size = 30;
    let edges = groups(1, size, 4, 0, 4);
    let n = size + 30;
    let y = forceatlas2_layout_2d_on_graph(n, &edges, ITERS, 1);
    assert!(all_finite(&y));
    let b = bbox(&y, &(0..n).collect::<Vec<_>>());
    let (w, h) = (b[1] - b[0], b[3] - b[2]);
    let aspect = w.max(h) / w.min(h);
    eprintln!("packing: aspect = {aspect}");
    assert!(aspect <= 3.0, "aspect {aspect}");
    let pieces: Vec<Vec<usize>> = std::iter::once((0..size).collect())
        .chain((size..n).map(|i| vec![i]))
        .collect();
    let boxes: Vec<[f32; 4]> = pieces.iter().map(|p| bbox(&y, p)).collect();
    let mut min_gap = f32::INFINITY;
    for p in 0..boxes.len() {
        for q in (p + 1)..boxes.len() {
            min_gap = min_gap.min(box_gap(&boxes[p], &boxes[q]));
        }
    }
    eprintln!("packing: min box gap = {min_gap}");
    assert!(min_gap > 0.0, "boxes touch");
    let y = forceatlas2_layout_2d_on_graph(25, &[], ITERS, 1);
    let b = bbox(&y, &(0..25).collect::<Vec<_>>());
    let (w, h) = (b[1] - b[0], b[3] - b[2]);
    let aspect = w.max(h) / w.min(h);
    eprintln!("packing, no edges: aspect = {aspect}");
    assert!(aspect <= 3.0, "aspect {aspect}");
}

/// Bar: shapes are right and coordinates finite for n = 0, 1, 2 (with and
/// without an edge); two nodes are placed apart.
#[test]
fn tiny_inputs() {
    assert_eq!(
        forceatlas2_layout_2d_on_graph(0, &[], ITERS, 0).shape(),
        (0, 2)
    );
    let y = forceatlas2_layout_2d_on_graph(1, &[], ITERS, 0);
    assert_eq!(y.shape(), (1, 2));
    assert!(all_finite(&y));
    for edges in [vec![], vec![(0, 1, 1.0)]] {
        let y = forceatlas2_layout_2d_on_graph(2, &edges, ITERS, 0);
        assert_eq!(y.shape(), (2, 2));
        assert!(all_finite(&y));
        assert!(dist(&y, 0, 1) > 0.0);
    }
    let y = forceatlas2_layout_2d_on_graph(3, &[(0, 1, 1.0), (1, 2, 1.0)], ITERS, 0);
    assert!(all_finite(&y));
}

/// Bar: the same seed gives a bitwise identical layout; another seed gives a
/// different one.
#[test]
fn same_seed_same_layout() {
    let edges = groups(3, 15, 3, 2, 9);
    let a = forceatlas2_layout_2d_on_graph(45, &edges, 100, 11);
    let b = forceatlas2_layout_2d_on_graph(45, &edges, 100, 11);
    let c = forceatlas2_layout_2d_on_graph(45, &edges, 100, 12);
    assert_eq!(a, b);
    assert_ne!(a, c);
}

/// Bar: a node with no edge is finite and its distance to the nearest other
/// node is at most half the diameter of the rest.
#[test]
fn lone_node_stays_near() {
    let n = 30;
    let y = forceatlas2_layout_2d_on_graph(n + 1, &chain(n), ITERS, 2);
    assert!(all_finite(&y));
    let near = (0..n).map(|i| dist(&y, i, n)).fold(f32::INFINITY, f32::min);
    let mut diam: f32 = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            diam = diam.max(dist(&y, i, j));
        }
    }
    eprintln!("lone node: nearest / diameter = {}", near / diam);
    assert!(diam > 0.0);
    assert!(near <= 0.5 * diam, "near {near} diam {diam}");
}

/// Bar: the layout has settled: going from 1000 to 1100 iterations moves the
/// nodes by a mean of at most 2% of the layout's diameter.
#[test]
fn layout_settles() {
    let size = 25;
    let edges = groups(3, size, 4, 3, 6);
    let n = 3 * size;
    let a = forceatlas2_layout_2d_on_graph(n, &edges, 1000, 4);
    let b = forceatlas2_layout_2d_on_graph(n, &edges, 1100, 4);
    let mut diam: f32 = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            diam = diam.max(dist(&b, i, j));
        }
    }
    let moved = (0..n)
        .map(|i| ((a[(i, 0)] - b[(i, 0)]).powi(2) + (a[(i, 1)] - b[(i, 1)]).powi(2)).sqrt())
        .sum::<f32>()
        / n as f32;
    eprintln!("settles: mean move / diameter = {}", moved / diam);
    assert!(diam > 0.0);
    assert!(moved <= 0.02 * diam, "moved {moved} diam {diam}");
}

/// Mean distance of `nodes` to their centroid.
fn spread(y: &Mat, nodes: &[usize]) -> f32 {
    let c = centroid(y, nodes);
    nodes
        .iter()
        .map(|&i| ((y[(i, 0)] - c[0]).powi(2) + (y[(i, 1)] - c[1]).powi(2)).sqrt())
        .sum::<f32>()
        / nodes.len() as f32
}

/// Bar: on a complete graph where only the weights tell three groups apart
/// (1 inside a group, 0.01 between), the closest two group centres are at
/// least 3 times the mean within-group spread apart.
#[test]
fn weights_alone_make_the_groups() {
    let size = 15;
    let n = 3 * size;
    let mut edges = vec![];
    for i in 0..n {
        for j in (i + 1)..n {
            edges.push((i, j, if i / size == j / size { 1.0 } else { 0.01 }));
        }
    }
    let y = forceatlas2_layout_2d_on_graph(n, &edges, ITERS, 8);
    let members: Vec<Vec<usize>> = (0..3)
        .map(|g| (g * size..(g + 1) * size).collect())
        .collect();
    let s = members.iter().map(|m| spread(&y, m)).sum::<f32>() / 3.0;
    let cents: Vec<[f32; 2]> = members.iter().map(|m| centroid(&y, m)).collect();
    let mut min_gap = f32::INFINITY;
    for a in 0..3 {
        for b in (a + 1)..3 {
            min_gap = min_gap.min(
                ((cents[a][0] - cents[b][0]).powi(2) + (cents[a][1] - cents[b][1]).powi(2)).sqrt(),
            );
        }
    }
    eprintln!("weights: min centre gap / spread = {}", min_gap / s);
    assert!(s > 0.0);
    assert!(min_gap >= 3.0 * s, "gap {min_gap} spread {s}");
}

/// Bar: two hubs joined by an edge, each with 15 leaves: the hubs are at
/// least 2 times the mean leaf-to-own-hub distance apart, so the two leaf
/// clouds do not run into each other.
#[test]
fn hubs_keep_their_leaves_apart() {
    let leaves = 15;
    let (h1, h2) = (0, 1);
    let mut edges = vec![(h1, h2, 1.0)];
    for k in 0..leaves {
        edges.push((h1, 2 + k, 1.0));
        edges.push((h2, 2 + leaves + k, 1.0));
    }
    let n = 2 + 2 * leaves;
    let y = forceatlas2_layout_2d_on_graph(n, &edges, ITERS, 5);
    let reach = (0..leaves)
        .map(|k| dist(&y, h1, 2 + k) + dist(&y, h2, 2 + leaves + k))
        .sum::<f32>()
        / (2 * leaves) as f32;
    let hubs = dist(&y, h1, h2);
    eprintln!("hubs: hub distance / leaf reach = {}", hubs / reach);
    assert!(hubs >= 2.0 * reach, "hubs {hubs} reach {reach}");
}

/// Bar: within a short budget of 150 steps, the chain of 40 already has
/// Spearman correlation at least 0.95 and the three groups a centre gap at
/// least 3 times their spread.
#[test]
fn converges_in_a_short_budget() {
    let n = 40;
    let y = forceatlas2_layout_2d_on_graph(n, &chain(n), 150, 3);
    let (mut a, mut b) = (vec![], vec![]);
    for i in 0..n {
        for j in (i + 1)..n {
            a.push((j - i) as f32);
            b.push(dist(&y, i, j));
        }
    }
    let rho = spearman(&a, &b);
    let size = 25;
    let y = forceatlas2_layout_2d_on_graph(3 * size, &groups(3, size, 4, 3, 1), 150, 7);
    let members: Vec<Vec<usize>> = (0..3)
        .map(|g| (g * size..(g + 1) * size).collect())
        .collect();
    let s = members.iter().map(|m| spread(&y, m)).sum::<f32>() / 3.0;
    let cents: Vec<[f32; 2]> = members.iter().map(|m| centroid(&y, m)).collect();
    let mut min_gap = f32::INFINITY;
    for p in 0..3 {
        for q in (p + 1)..3 {
            min_gap = min_gap.min(
                ((cents[p][0] - cents[q][0]).powi(2) + (cents[p][1] - cents[q][1]).powi(2)).sqrt(),
            );
        }
    }
    eprintln!(
        "short budget: chain rho = {rho}, groups gap / spread = {}",
        min_gap / s
    );
    assert!(rho >= 0.95, "rho {rho}");
    assert!(min_gap >= 3.0 * s, "gap {min_gap} spread {s}");
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
        forceatlas2_layout_2d_on_graph(6, &edges, 50, 3),
        forceatlas2_layout_2d_on_graph(6, &kept, 50, 3)
    );
}
