use super::*;
use crate::matrix::agreement::spearman;
use crate::matrix::test_support::{fuzzy_edges, pca_start};
use crate::matrix::umap::Umap;
use rand::{rngs::SmallRng, RngExt, SeedableRng};
use std::f32::consts::PI;

/// A Swiss roll `(t cos t, h, t sin t)`, `t` in `[1.5π, 4.5π]` and `h` in
/// `[0, 30]`, embedded in 50 dimensions with a little noise. Returns the
/// data and each point's place on the unrolled sheet (arc length along the
/// spiral, height). Its two leading PCs are the height and one side of the
/// spiral, so the PCA start lays the turns on top of one another.
pub(crate) fn swiss_roll(n: usize, seed: u64) -> (Mat, Vec<[f32; 2]>) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let basis = Mat::from_fn(3, 50, |_, _| rng.random_range(-1.0_f32..1.0));
    let mut roll = Mat::zeros(n, 3);
    let mut sheet = Vec::with_capacity(n);
    for i in 0..n {
        let t = rng.random_range(1.5 * PI..4.5 * PI);
        let h = rng.random_range(0.0..30.0);
        roll[(i, 0)] = t * t.cos();
        roll[(i, 1)] = h;
        roll[(i, 2)] = t * t.sin();
        let arc = 0.5 * (t * (1.0 + t * t).sqrt() + t.asinh());
        sheet.push([arc, h]);
    }
    let noise = Mat::from_fn(n, 50, |_, _| rng.random_range(-0.1_f32..0.1));
    (roll * basis + noise, sheet)
}

/// A Swiss roll, its kNN graph and its landmark start.
pub(crate) struct Roll {
    pub data: Mat,
    pub sheet: Vec<[f32; 2]>,
    pub edges: Vec<(usize, usize, f32)>,
    pub start: Vec<f32>,
}

pub(crate) fn roll_from_landmarks(n: usize, seed: u64) -> Roll {
    let (data, sheet) = swiss_roll(n, seed);
    let edges = fuzzy_edges(&data, 15);
    let start = landmark_init_2d(&data, &edges, N_LANDMARKS, 6);
    Roll {
        data,
        sheet,
        edges,
        start,
    }
}

fn dist2d(y: &[f32], i: usize, j: usize) -> f32 {
    (y[2 * i] - y[2 * j]).hypot(y[2 * i + 1] - y[2 * j + 1])
}

/// Rank correlation between distances on the unrolled sheet and in the
/// layout `y`, over random pairs: near 1 when the layout unrolls the roll.
pub(crate) fn unrolled(sheet: &[[f32; 2]], y: &[f32]) -> f32 {
    let mut rng = SmallRng::seed_from_u64(11);
    let n = sheet.len();
    let (mut on_sheet, mut in_layout) = (vec![], vec![]);
    for _ in 0..20_000 {
        let (i, j) = (rng.random_range(0..n), rng.random_range(0..n));
        on_sheet.push(dist2d(sheet.as_flattened(), i, j));
        in_layout.push(dist2d(y, i, j));
    }
    spearman(&on_sheet, &in_layout)
}

/// Panics if two points of the layout `y` start on the same spot.
fn assert_no_coincident(y: &[f32]) {
    for i in 0..y.len() / 2 {
        for j in 0..i {
            assert!(dist2d(y, i, j) > 0.0, "points {i} and {j} coincide");
        }
    }
}

/// `copies` of each row of `base` in turn, each copy linked to the next
/// copy of its row.
fn repeated(base: &[[f32; 3]], copies: usize) -> (Mat, Vec<(usize, usize, f32)>) {
    let (r, n) = (base.len(), base.len() * copies);
    let data = Mat::from_fn(n, 3, |i, c| base[i % r][c]);
    let edges = (0..n - r).map(|i| (i, i + r, 1.0)).collect();
    (data, edges)
}

#[test]
fn a_swiss_roll_starts_unrolled() {
    let roll = roll_from_landmarks(2000, 1);
    let pca = unrolled(&roll.sheet, &pca_start(&roll.data));
    let landmarks = unrolled(&roll.sheet, &roll.start);
    assert!(pca < 0.5, "the PCA start should fold the roll: {pca}");
    assert!(landmarks > 0.95, "landmark start {landmarks}");
}

#[test]
fn a_few_shortcut_edges_do_not_fold_it() {
    // Edges between far parts of the roll, as noisy kNN graphs have: one
    // landmark pair joined by a single edge must not become a path.
    let n = 2000;
    let (data, sheet) = swiss_roll(n, 2);
    let mut edges = fuzzy_edges(&data, 15);
    let mut rng = SmallRng::seed_from_u64(2);
    for _ in 0..n / 100 {
        let (i, j) = (rng.random_range(0..n), rng.random_range(0..n));
        if i != j {
            edges.push((i.min(j), i.max(j), 0.5));
        }
    }
    let start = unrolled(&sheet, &landmark_init_2d(&data, &edges, N_LANDMARKS, 6));
    assert!(start > 0.95, "landmark start {start}");
}

/// Six groups of five Gaussian clusters in 50 dimensions, with labels: no
/// kNN edge joins two clusters.
pub(crate) fn clusters(n: usize, seed: u64) -> (Mat, Vec<usize>) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let groups = Mat::from_fn(6, 50, |_, _| rng.random_range(-6.0_f32..6.0));
    let centres = Mat::from_fn(30, 50, |r, c| {
        groups[(r / 5, c)] + rng.random_range(-2.0_f32..2.0)
    });
    let labels: Vec<usize> = (0..n).map(|_| rng.random_range(0..30)).collect();
    let data = Mat::from_fn(n, 50, |i, c| {
        centres[(labels[i], c)] + rng.random_range(-0.6_f32..0.6)
    });
    (data, labels)
}

/// The mean row of `m` for each label.
fn label_means(m: &Mat, labels: &[usize]) -> Mat {
    let k = labels.iter().max().unwrap() + 1;
    let mut sum = Mat::zeros(k, m.ncols());
    let mut size = vec![0.0_f32; k];
    for (i, &l) in labels.iter().enumerate() {
        let mut row = sum.row_mut(l);
        row += m.row(i);
        size[l] += 1.0;
    }
    for (l, &s) in size.iter().enumerate() {
        sum.row_mut(l).unscale_mut(s);
    }
    sum
}

/// Distances between the rows of `m`, each pair once.
fn pair_distances(m: &Mat) -> Vec<f32> {
    let k = m.nrows();
    (0..k)
        .flat_map(|p| (p + 1..k).map(move |q| (p, q)))
        .map(|(p, q)| (m.row(p) - m.row(q)).norm())
        .collect()
}

/// Rank correlation between the distances of the label means in `data`
/// and in the layout `y`.
pub(crate) fn arranged(data: &Mat, labels: &[usize], y: &[f32]) -> f32 {
    let y = Mat::from_row_slice(labels.len(), 2, y);
    spearman(
        &pair_distances(&label_means(data, labels)),
        &pair_distances(&label_means(&y, labels)),
    )
}

#[test]
fn separate_clusters_start_arranged_as_their_pca() {
    // Each landmark weighs as many points as it stands for, so on clusters
    // no edge joins, the start is close to the PCA of the points, not to
    // that of the landmarks.
    let (data, labels) = clusters(3000, 21);
    let edges = fuzzy_edges(&data, 15);
    let pca = arranged(&data, &labels, &pca_start(&data));
    let landmarks = arranged(
        &data,
        &labels,
        &landmark_init_2d(&data, &edges, N_LANDMARKS, 6),
    );
    assert!(landmarks > pca - 0.01, "landmarks {landmarks} vs PCA {pca}");
}

#[test]
fn the_fit_keeps_the_roll_unrolled() {
    let n = 1500;
    let Roll {
        data,
        sheet,
        edges,
        start,
    } = roll_from_landmarks(n, 3);
    let umap = Umap {
        n_epochs: 300,
        ..Umap::default()
    };
    let from_pca = unrolled(&sheet, &umap.fit(&edges, n, &pca_start(&data)));
    let from_landmarks = unrolled(&sheet, &umap.fit(&edges, n, &start));
    assert!(
        from_landmarks > 0.9 && from_landmarks > from_pca + 0.2,
        "from landmarks {from_landmarks}, from PCA {from_pca}"
    );
}

#[test]
fn the_start_is_seeded_scaled_and_has_no_coincident_points() {
    // Each of 4 points repeated 50 times: points of one landmark must still
    // start apart.
    let base = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]];
    let (data, edges) = repeated(&base, 50);
    let y = landmark_init_2d(&data, &edges, N_LANDMARKS, 1);
    assert_eq!(y, landmark_init_2d(&data, &edges, N_LANDMARKS, 1));
    assert_ne!(y, landmark_init_2d(&data, &edges, N_LANDMARKS, 2));
    let reach = y.iter().fold(0.0_f32, |m, &v| m.max(v.abs()));
    assert!(
        (0.5 * INIT_SCALE..=1.5 * INIT_SCALE).contains(&reach),
        "{reach}"
    );
    assert_no_coincident(&y);
}

#[test]
fn tiny_inputs_get_a_start_of_their_size() {
    for n in [0, 1, 2, 5] {
        let data = Mat::from_fn(n, 3, |i, c| (i * 3 + c) as f32);
        let edges: Vec<(usize, usize, f32)> = (1..n).map(|i| (i - 1, i, 1.0)).collect();
        let y = landmark_init_2d(&data, &edges, N_LANDMARKS, 1);
        assert_eq!(y.len(), 2 * n);
        assert!(y.iter().all(|v| v.is_finite()), "n {n}: {y:?}");
    }
}

#[test]
fn landmarks_the_plane_lays_on_one_another_still_spread_their_points() {
    // The corners of a 4 x 3 x 2 box: the 2D MDS keeps its two longer
    // sides, so each corner lands on its twin across the shortest.
    let corners: Vec<[f32; 3]> = (0..8)
        .map(|c| {
            let side = |bit: usize, half: f32| if c >> bit & 1 == 1 { half } else { -half };
            [side(0, 2.0), side(1, 1.5), side(2, 1.0)]
        })
        .collect();
    let (data, edges) = repeated(&corners, 50);
    assert_no_coincident(&landmark_init_2d(&data, &edges, N_LANDMARKS, 1));
}
