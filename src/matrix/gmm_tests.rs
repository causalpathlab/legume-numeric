use super::*;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

/// `n_per` rows about each of `centres` (rows of `d`), Gaussian noise of SD
/// `sd`, in centre order.
fn blobs(centres: &[&[f32]], n_per: usize, sd: f32, seed: u64) -> DMatrix<f32> {
    let d = centres[0].len();
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    let noise = Normal::new(0.0f32, sd).unwrap();
    DMatrix::from_fn(centres.len() * n_per, d, |i, j| {
        centres[i / n_per][j] + noise.sample(&mut rng)
    })
}

#[test]
fn bic_finds_the_planted_blobs_and_one_for_one_gaussian() {
    let z = blobs(&[&[0.0, 0.0], &[6.0, 0.0], &[0.0, 6.0]], 200, 1.0, 1);
    let c = choose_gmm(&z, 6, Criterion::Bic, 3, 42, 200);
    assert_eq!(c.fit.model.k(), 3, "BIC by k: {:?}", c.score);
    assert_eq!(c.score.len(), 6);
    let one = blobs(&[&[0.0, 0.0]], 600, 1.0, 2);
    assert_eq!(
        choose_gmm(&one, 6, Criterion::Bic, 3, 42, 200)
            .fit
            .model
            .k(),
        1
    );
}

#[test]
fn a_noisy_dimension_does_not_split_clumps_a_tight_one_separates() {
    // Dimension 0: two tight clumps; dimension 1: wide Gaussian noise.
    let z = blobs(&[&[-1.0, 0.0], &[1.0, 0.0]], 300, 0.05, 3);
    let mut z = z;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(4);
    let wide = Normal::new(0.0f32, 1.0).unwrap();
    for i in 0..z.nrows() {
        z[(i, 1)] = wide.sample(&mut rng);
    }
    let c = choose_gmm(&z, 6, Criterion::Bic, 3, 42, 200);
    assert_eq!(c.fit.model.k(), 2, "BIC by k: {:?}", c.score);
    // Each clump is one component.
    assert!(c.fit.labels[..300].iter().all(|&l| l == c.fit.labels[0]));
    assert!(c.fit.labels[300..].iter().all(|&l| l != c.fit.labels[0]));
    // The recorded model assigns the same rows the same way, also after a
    // trip through JSON.
    let back: DiagGmm =
        serde_json::from_str(&serde_json::to_string(&c.fit.model).unwrap()).unwrap();
    assert_eq!(back.assign(&z), c.fit.labels);
}

#[test]
fn a_fit_is_the_same_on_any_thread_count() {
    let z = blobs(&[&[0.0], &[5.0]], 5000, 1.0, 5);
    let fit = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| fit_diag_gmm(&z, &kmeans_start(&z, 2, 2, 7), 100))
    };
    let (a, b) = (fit(1), fit(4));
    assert_eq!(a.labels, b.labels);
    assert_eq!(a.model, b.model);
}

#[test]
fn icl_keeps_one_skewed_hump_whole_where_bic_cuts_it() {
    // A skewed single hump: exp of a Gaussian, 4000 rows.
    let mut rng = rand::rngs::SmallRng::seed_from_u64(9);
    let g = Normal::new(0.0f32, 0.6).unwrap();
    let z = DMatrix::from_fn(4000, 1, |_, _| g.sample(&mut rng).exp());
    let icl = choose_gmm(&z, 6, Criterion::Icl, 3, 42, 200);
    assert_eq!(icl.fit.model.k(), 1, "ICL by k: {:?}", icl.score);
    let bic = choose_gmm(&z, 6, Criterion::Bic, 3, 42, 200);
    assert!(bic.fit.model.k() > 1, "BIC by k: {:?}", bic.score);
    // Separated blobs are still found.
    let z = blobs(&[&[0.0, 0.0], &[6.0, 0.0], &[0.0, 6.0]], 200, 1.0, 1);
    assert_eq!(
        choose_gmm(&z, 6, Criterion::Icl, 3, 42, 200).fit.model.k(),
        3
    );
}
