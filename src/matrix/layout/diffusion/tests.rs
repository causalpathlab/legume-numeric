use super::*;
use rand::{rngs::SmallRng, RngExt, SeedableRng};

fn chain(n: usize) -> Vec<(usize, usize, f32)> {
    (0..n - 1).map(|i| (i, i + 1, 1.0)).collect()
}

fn ring(n: usize) -> Vec<(usize, usize, f32)> {
    (0..n)
        .map(|i| (i.min((i + 1) % n), i.max((i + 1) % n), 1.0))
        .collect()
}

fn bounding_box(y: &Mat, nodes: &[usize]) -> [f32; 4] {
    let mut b = [
        f32::INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
    ];
    for &i in nodes {
        b[0] = b[0].min(y[(i, 0)]);
        b[1] = b[1].min(y[(i, 1)]);
        b[2] = b[2].max(y[(i, 0)]);
        b[3] = b[3].max(y[(i, 1)]);
    }
    b
}

/// Bar: along a chain with uneven weights, the first component is strictly
/// monotone in the chain's order, for t = 0 and t = 3.
#[test]
fn chain_first_component_is_monotone() {
    let n = 30;
    let edges: Vec<_> = (0..n - 1)
        .map(|i| (i, i + 1, 0.5 + (i % 4) as f32 * 0.4))
        .collect();
    for t in [0, 3] {
        let y = diffusion_map_2d_on_graph(n, &edges, t);
        let steps: Vec<f32> = (0..n - 1).map(|i| y[(i + 1, 0)] - y[(i, 0)]).collect();
        let up = steps.iter().all(|&s| s > 0.0);
        let down = steps.iter().all(|&s| s < 0.0);
        assert!(up || down, "t = {t}: steps {steps:?}");
    }
}

/// Bar: a ring is laid out as a circle, radii within 1e-3 of their mean
/// (relative), and the angle moves the same way around at every step,
/// wrapping exactly once.
#[test]
fn ring_is_a_circle() {
    let n = 40;
    let y = diffusion_map_2d_on_graph(n, &ring(n), 2);
    let (cx, cy) = (y.column(0).mean(), y.column(1).mean());
    let radii: Vec<f32> = (0..n)
        .map(|i| (y[(i, 0)] - cx).hypot(y[(i, 1)] - cy))
        .collect();
    let mean = radii.iter().sum::<f32>() / n as f32;
    let spread = radii.iter().map(|r| (r - mean).abs()).fold(0.0, f32::max) / mean;
    eprintln!("ring: max relative radius deviation {spread:.2e}");
    assert!(spread < 1e-3, "radius deviation {spread}");

    let angle = |i: usize| (y[(i, 1)] - cy).atan2(y[(i, 0)] - cx);
    let turns: Vec<f32> = (0..n)
        .map(|i| {
            let d = angle((i + 1) % n) - angle(i);
            (d + std::f32::consts::PI).rem_euclid(2.0 * std::f32::consts::PI) - std::f32::consts::PI
        })
        .collect();
    let same_way = turns.iter().all(|&d| d > 0.0) || turns.iter().all(|&d| d < 0.0);
    let total: f32 = turns.iter().sum::<f32>().abs();
    eprintln!(
        "ring: total turn {total:.4} (2 pi = {:.4})",
        2.0 * std::f32::consts::PI
    );
    assert!(same_way, "turns {turns:?}");
    assert!((total - 2.0 * std::f32::consts::PI).abs() < 1e-3);
}

/// Bar: three dense groups linked by a few weak edges are laid out apart:
/// the closest pair of group centres is at least 5 times farther apart
/// than the widest group's RMS radius.
#[test]
fn weakly_linked_groups_stay_apart() {
    let (groups, per) = (3, 20);
    let n = groups * per;
    let mut rng = SmallRng::seed_from_u64(7);
    let mut edges = vec![];
    for g in 0..groups {
        for a in 0..per {
            for b in (a + 1)..per {
                if rng.random_range(0.0_f32..1.0) < 0.5 {
                    edges.push((g * per + a, g * per + b, rng.random_range(0.5_f32..1.0)));
                }
            }
        }
    }
    for g in 0..groups {
        let h = (g + 1) % groups;
        for k in 0..3 {
            let (a, b) = (g * per + k, h * per + k + 5);
            edges.push((a.min(b), a.max(b), 0.01));
        }
    }
    let y = diffusion_map_2d_on_graph(n, &edges, 1);
    let centre = |g: usize| {
        let r = (g * per)..((g + 1) * per);
        let cx = r.clone().map(|i| y[(i, 0)]).sum::<f32>() / per as f32;
        let cy = r.map(|i| y[(i, 1)]).sum::<f32>() / per as f32;
        (cx, cy)
    };
    let mut widest = 0.0_f32;
    for g in 0..groups {
        let (cx, cy) = centre(g);
        let ms = ((g * per)..((g + 1) * per))
            .map(|i| (y[(i, 0)] - cx).powi(2) + (y[(i, 1)] - cy).powi(2))
            .sum::<f32>()
            / per as f32;
        widest = widest.max(ms.sqrt());
    }
    let mut closest = f32::INFINITY;
    for g in 0..groups {
        for h in (g + 1)..groups {
            let ((a, b), (c, d)) = (centre(g), centre(h));
            closest = closest.min((a - c).hypot(b - d));
        }
    }
    eprintln!(
        "groups: closest centres / widest radius = {:.1}",
        closest / widest
    );
    assert!(closest > 5.0 * widest, "closest {closest}, widest {widest}");
}

/// Bar: a graph in four pieces (a chain, a ring, a pair, a lone node) gives
/// pieces whose bounding boxes are pairwise apart by at least 1% of the
/// wider box, and each piece is laid
/// out as it is alone, up to a shift (max deviation 1e-5 of its extent).
#[test]
fn separate_pieces_do_not_overlap() {
    let pieces: Vec<Vec<usize>> = vec![
        (0..12).collect(),
        (12..32).collect(),
        vec![32, 33],
        vec![34],
    ];
    let mut edges = chain(12);
    edges.extend(ring(20).into_iter().map(|(i, j, w)| (i + 12, j + 12, w)));
    edges.push((32, 33, 2.0));
    let n = 35;
    let y = diffusion_map_2d_on_graph(n, &edges, 1);
    assert!(y.iter().all(|v| v.is_finite()));

    let boxes: Vec<[f32; 4]> = pieces.iter().map(|p| bounding_box(&y, p)).collect();
    for a in 0..boxes.len() {
        for b in (a + 1)..boxes.len() {
            let (p, q) = (boxes[a], boxes[b]);
            let margin = 0.01 * (p[2] - p[0]).max(q[2] - q[0]);
            let apart = p[2] + margin < q[0]
                || q[2] + margin < p[0]
                || p[3] + margin < q[1]
                || q[3] + margin < p[1];
            assert!(apart, "pieces {a} and {b} overlap: {p:?} {q:?}");
        }
    }

    let alone = [
        diffusion_map_2d_on_graph(12, &chain(12), 1),
        diffusion_map_2d_on_graph(20, &ring(20), 1),
    ];
    for (piece, solo) in pieces.iter().zip(&alone) {
        let extent = solo.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
        for dim in 0..2 {
            let shift = y[(piece[0], dim)] - solo[(0, dim)];
            for (k, &i) in piece.iter().enumerate() {
                let dev = (y[(i, dim)] - shift - solo[(k, dim)]).abs();
                assert!(dev <= 1e-5 * extent, "piece node {i}: deviation {dev}");
            }
        }
    }
}

/// Bar: n = 0, 1, 2 and 3, with and without edges, give n x 2 finite
/// coordinates; two linked nodes are placed apart, and so are three nodes
/// with no edges (each its own piece).
#[test]
fn tiny_inputs() {
    assert_eq!(diffusion_map_2d_on_graph(0, &[], 0).shape(), (0, 2));
    for (n, edges) in [
        (1, vec![]),
        (1, vec![(0, 0, 1.0)]),
        (2, vec![]),
        (2, vec![(0, 1, 1.0)]),
        (3, vec![]),
        (3, vec![(0, 1, 1.0), (1, 2, 1.0)]),
        (3, vec![(0, 1, 0.0)]),
    ] {
        let y = diffusion_map_2d_on_graph(n, &edges, 1);
        assert_eq!(y.shape(), (n, 2));
        assert!(y.iter().all(|v| v.is_finite()), "n = {n}, {edges:?}: {y}");
    }
    let apart = |y: &Mat, i: usize, j: usize| (y[(i, 0)] - y[(j, 0)]).hypot(y[(i, 1)] - y[(j, 1)]);
    let y = diffusion_map_2d_on_graph(2, &[(0, 1, 1.0)], 0);
    assert!(apart(&y, 0, 1) > 0.0);
    let y = diffusion_map_2d_on_graph(3, &[], 0);
    assert!(apart(&y, 0, 1) > 0.0 && apart(&y, 0, 2) > 0.0 && apart(&y, 1, 2) > 0.0);
}

/// Bar: on a chain the second axis is a higher harmonic and decays faster,
/// so its spread relative to the first shrinks with t by exactly
/// (lambda_2 / lambda_1)^t, within 1e-3 relative.
#[test]
fn diffusion_time_scales_by_eigenvalues() {
    let n = 25;
    let spread = |y: &Mat, d: usize| y.column(d).iter().map(|v| v * v).sum::<f32>().sqrt();
    let y0 = diffusion_map_2d_on_graph(n, &chain(n), 0);
    let y4 = diffusion_map_2d_on_graph(n, &chain(n), 4);
    // A chain's normalized operator has eigenvalues cos(pi k / (n - 1)).
    let l = |k: usize| (std::f32::consts::PI * k as f32 / (n - 1) as f32).cos();
    for (d, k) in [(0, 1), (1, 2)] {
        let ratio = spread(&y4, d) / spread(&y0, d);
        let want = l(k).powi(4);
        eprintln!("t scaling, axis {d}: {ratio:.6} vs {want:.6}");
        assert!((ratio - want).abs() < 1e-3 * want);
    }
}
