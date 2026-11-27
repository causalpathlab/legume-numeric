use super::*;

/// Seeded noise of SD about 0.2, near normal (a sum of uniforms): X-means
/// splits uniform noise, which no score here is.
fn jitter(i: usize, j: usize) -> f32 {
    use rand::{RngExt, SeedableRng};
    let mut rng = rand::rngs::SmallRng::seed_from_u64((i * 131 + j) as u64);
    (0..6).map(|_| rng.random::<f32>() - 0.5).sum::<f32>() * 0.28
}

/// A projection with no structure of its own: one group of like cells.
fn flat_projection(n: usize) -> DMatrix<f32> {
    DMatrix::from_fn(4, n, |d, i| ((i * 7 + d * 13) % 11) as f32)
}

#[test]
fn two_clumps_within_each_batch_become_two_bins_and_assign_agrees() {
    // Two batches, 200 cells each; score 0 has two clumps in both, score 1
    // is noise of the same covariate, score 2 another covariate's noise.
    let n = 400;
    let batch: Vec<Box<str>> = (0..n)
        .map(|i| if i < 200 { "a" } else { "b" }.into())
        .collect();
    let scores = DMatrix::from_fn(n, 3, |i, j| {
        let clump = if j == 0 && i % 2 == 0 { 4.0 } else { 0.0 };
        clump + jitter(i, j)
    });
    let cov: Vec<Box<str>> = ["x", "x", "y"].map(Box::from).to_vec();
    let (bin, m) = fit_bins(
        &scores,
        &cov,
        &batch,
        &flat_projection(n),
        &BinOpts::default(),
    )
    .unwrap()
    .expect("binned");
    assert_eq!(m.mixture.k(), 2);
    assert_eq!(m.own_k["x"], 2);
    assert_eq!(m.own_k["y"], 1);
    assert_eq!(m.set_aside, [Box::from("y")]);
    assert_eq!(m.binned, [0, 1]);
    assert_ne!(bin[0], bin[1]);
    assert_eq!(bin[0], bin[2]);
    // The recorded fit puts the same cells in the same bins.
    assert_eq!(m.assign(&scores, &batch), bin);
    // It serialises.
    let back: CovariateBins = serde_json::from_str(&serde_json::to_string(&m).unwrap()).unwrap();
    assert_eq!(back.assign(&scores, &batch), bin);
}

#[test]
fn a_shift_between_batches_alone_is_not_binned() {
    let n = 400;
    let batch: Vec<Box<str>> = (0..n)
        .map(|i| if i < 200 { "a" } else { "b" }.into())
        .collect();
    // Batch b sits 6 higher; within each batch, noise only.
    let scores = DMatrix::from_fn(n, 1, |i, j| if i < 200 { 0.0 } else { 6.0 } + jitter(i, j));
    let cov = vec![Box::from("sex")];
    let r = fit_bins(
        &scores,
        &cov,
        &batch,
        &flat_projection(n),
        &BinOpts::default(),
    )
    .unwrap();
    assert_eq!(
        r.err(),
        Some(NotBinned::NoStructure {
            set_aside: vec![Box::from("sex")]
        })
    );
}

#[test]
fn a_sparse_bin_merges_into_its_nearest_and_a_lone_bin_is_none() {
    let centroids = DMatrix::from_row_slice(3, 1, &[0.0, 1.0, 10.0]);
    // Batch a: 60 in bin 0, 5 in bin 1 (merges into 0), 60 in bin 2.
    // Batch b: 30 in bin 2 only (no bin).
    let mut bin: Vec<Option<usize>> = std::iter::repeat_n(Some(0), 60)
        .chain(std::iter::repeat_n(Some(1), 5))
        .chain(std::iter::repeat_n(Some(2), 60))
        .chain(std::iter::repeat_n(Some(2), 30))
        .collect();
    let batch: Vec<Box<str>> = (0..155)
        .map(|i| if i < 125 { "a" } else { "b" }.into())
        .collect();
    let remap = merge_sparse_levels(&mut bin, &cells_by_batch(&batch), &centroids, 10);
    assert!(bin[60..65].iter().all(|&c| c == Some(0)));
    assert_eq!(bin[65], Some(2));
    assert!(bin[125..].iter().all(Option::is_none));
    assert_eq!(remap["a"], [Some(0), Some(0), Some(2)]);
    assert_eq!(remap["b"], [None, None, None]);
}

#[test]
fn a_group_wholly_in_one_bin_keeps_no_bin() {
    let mut bin: Vec<Option<usize>> = (0..40)
        .map(|i| Some(usize::from(i >= 20 && i % 2 == 0)))
        .collect();
    let groups: Vec<usize> = (0..40).map(|i| usize::from(i >= 20)).collect();
    assert_eq!(clear_one_bin_groups(&mut bin, &groups, 2, 0.95), 1);
    assert!(bin[..20].iter().all(Option::is_none));
    assert!(bin[20..].iter().all(Option::is_some));
}

#[test]
fn sizes_that_disagree_are_refused() {
    let scores = DMatrix::<f32>::zeros(10, 2);
    let batch: Vec<Box<str>> = vec!["a".into(); 10];
    assert!(fit_bins(
        &scores,
        &[Box::from("x")],
        &batch,
        &flat_projection(10),
        &BinOpts::default()
    )
    .is_err());
}
