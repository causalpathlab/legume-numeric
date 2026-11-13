use super::*;
use crate::matrix::test_support::fuzzy_edges;
use nalgebra::DMatrix;
use rand::{rngs::SmallRng, RngExt, SeedableRng};

/// `n` points in 10 dimensions around each of `blobs` centres far apart,
/// with the blob of each point.
fn blobs(n: usize, blobs: usize, seed: u64) -> (DMatrix<f32>, Vec<usize>) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let blob: Vec<usize> = (0..n).map(|i| i % blobs).collect();
    let data = DMatrix::from_fn(n, 10, |i, c| {
        let centre = if c == blob[i] % 10 {
            20.0 * (1 + blob[i] / 10) as f32
        } else {
            0.0
        };
        centre + rng.random_range(-1.0_f32..1.0)
    });
    (data, blob)
}

/// Each group's members, by group id.
fn members(membership: &[usize]) -> Vec<Vec<usize>> {
    let n_groups = membership.iter().max().map_or(0, |&g| g + 1);
    let mut out = vec![Vec::new(); n_groups];
    for (i, &g) in membership.iter().enumerate() {
        out[g].push(i);
    }
    out
}

#[test]
fn groups_do_not_cross_a_weak_link() {
    // Two blobs whose kNN graph is joined by a few weak edges, as noisy
    // graphs are: a group is a stretch of one blob, never a bit of each.
    let n = 1000;
    let (data, blob) = blobs(n, 2, 1);
    let mut edges = fuzzy_edges(&data, 15);
    let mut rng = SmallRng::seed_from_u64(1);
    for _ in 0..40 {
        let (i, j) = (
            2 * rng.random_range(0..n / 2),
            2 * rng.random_range(0..n / 2) + 1,
        );
        edges.push((i.min(j), i.max(j), 0.2));
    }
    let membership = coarsen(n, &edges, 50, 7);
    for (g, idx) in members(&membership).iter().enumerate() {
        let first = blob[idx[0]];
        assert!(
            idx.iter().all(|&i| blob[i] == first),
            "group {g} spans both blobs"
        );
    }
}

#[test]
fn the_target_is_met_with_groups_of_even_size() {
    let n = 1000;
    let (data, _) = blobs(n, 1, 2);
    let membership = coarsen(n, &fuzzy_edges(&data, 15), 50, 7);
    let groups = members(&membership);
    assert_eq!(groups.len(), 50);
    // Within a factor of two or three of the mean size, 20.
    let largest = groups.iter().map(Vec::len).max().unwrap();
    let smallest = groups.iter().map(Vec::len).min().unwrap();
    assert!(largest <= 40, "a group of {largest}");
    assert!(smallest >= 7, "a group of {smallest}");
}

#[test]
fn points_with_no_edge_between_them_never_share_a_group() {
    // Five points and one edge: at most one merge is possible, so four
    // groups are as few as there can be.
    let edges = [(0, 1, 1.0)];
    let membership = coarsen(5, &edges, 2, 1);
    assert_eq!(members(&membership).len(), 4);
    assert_eq!(membership[0], membership[1]);
}

#[test]
fn the_same_seed_gives_the_same_groups() {
    let (data, _) = blobs(400, 3, 3);
    let edges = fuzzy_edges(&data, 15);
    assert_eq!(coarsen(400, &edges, 30, 5), coarsen(400, &edges, 30, 5));
}

#[test]
fn a_target_at_or_above_the_point_count_keeps_every_point() {
    let edges = [(0, 1, 1.0), (1, 2, 1.0)];
    assert_eq!(coarsen(3, &edges, 3, 1), vec![0, 1, 2]);
    assert_eq!(coarsen(3, &edges, 10, 1), vec![0, 1, 2]);
    assert!(coarsen(0, &[], 4, 1).is_empty());
}

#[test]
fn collapsing_sums_the_weights_between_groups() {
    // Groups {0, 1} and {2, 3}, with node 4 alone.
    let edges = [
        (0, 1, 5.0), // inside a group: dropped
        (0, 2, 1.0),
        (1, 2, 2.0),
        (1, 3, 0.5),
        (3, 4, 4.0),
    ];
    let membership = [0, 0, 1, 1, 2];
    assert_eq!(
        collapse(&edges, &membership),
        vec![(0, 1, 3.5), (1, 2, 4.0)]
    );
    // A node in no group takes its edges with it.
    let membership = [0, 0, 1, 1, usize::MAX];
    assert_eq!(collapse(&edges, &membership), vec![(0, 1, 3.5)]);
}

#[test]
fn the_strongest_link_is_merged_first() {
    // A chain of strong pairs joined by links still strong enough to take:
    // whatever the visiting order, the pairs are the strong ones.
    let edges: Vec<(usize, usize, f32)> = (0..7)
        .map(|i| (i, i + 1, if i % 2 == 0 { 1.0 } else { 0.6 }))
        .collect();
    for seed in 0..20 {
        let membership = coarsen(8, &edges, 4, seed);
        for i in (0..8).step_by(2) {
            assert_eq!(
                membership[i],
                membership[i + 1],
                "seed {seed}: {membership:?}"
            );
        }
    }
}

#[test]
fn a_hub_does_not_hold_the_rounds_back() {
    // Every spoke's strongest link is the hub, which only one of them can
    // take a round: the rounds must still finish in time.
    let n = 20_001;
    let edges: Vec<(usize, usize, f32)> = (1..n).map(|i| (0, i, 1.0)).collect();
    let start = std::time::Instant::now();
    let membership = coarsen(n, &edges, 10, 1);
    assert!(start.elapsed().as_secs_f32() < 5.0, "{:?}", start.elapsed());
    assert_eq!(membership.len(), n);
}
