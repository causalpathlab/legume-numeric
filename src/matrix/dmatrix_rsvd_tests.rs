use super::*;
use crate::matrix::traits::{RandomizedAlgs, RsvdArgs, SampleOps};

/// A planted rank-one spike over unit noise, well above the noise edge:
/// the randomised SVD must recover its singular value and direction.
fn spiked(n: usize, p: usize, strength: f32, seed: u64) -> (DMatrix<f32>, DVector<f32>) {
    let noise = DMatrix::<f32>::rnorm_seeded(n, p, seed);
    let mut u = DVector::<f32>::from_fn(n, |i, _| if i % 2 == 0 { 1.0 } else { -1.0 });
    u /= u.norm();
    let mut v = DVector::<f32>::from_fn(p, |j, _| if j < p / 3 { 1.0 } else { 0.0 });
    v /= v.norm();
    (noise + strength * (&u * v.transpose()), v)
}

#[test]
fn rsvd_recovers_a_planted_spike() {
    let (x, v) = spiked(120, 200, 40.0, 1);
    let full = x.clone().svd(false, true);
    let exact = full.singular_values[0];
    let exact_v = full.v_t.as_ref().unwrap().row(0).transpose();
    let (_, s, vv) = x.rsvd(2).unwrap();
    let s1 = s[0];
    assert!(
        (s1 - exact).abs() <= 0.02 * exact,
        "leading singular value {s1} vs exact {exact}"
    );
    let cos_exact = vv.column(0).dot(&exact_v).abs();
    assert!(
        cos_exact >= 0.99,
        "cosine with the exact leading vector {cos_exact}"
    );
    // and the exact vector itself is the plant up to noise tilt
    let cos_plant = exact_v.dot(&v).abs();
    assert!(cos_plant >= 0.85, "exact vs planted cosine {cos_plant}");
}

#[test]
fn rsvd_leading_value_of_noise_stays_at_the_edge() {
    let x = DMatrix::<f32>::rnorm_seeded(120, 200, 2);
    let exact = x.clone().svd(false, false).singular_values[0];
    let (_, s, _) = x.rsvd(2).unwrap();
    assert!(
        s[0] <= exact * 1.001,
        "randomised value cannot exceed the exact one"
    );
    assert!(
        s[0] >= 0.9 * exact,
        "randomised value {} far below exact {exact}",
        s[0]
    );
}

/// A symmetric matrix whose leading eigenvalues sit close together, as a
/// diffusion operator's do: `Q diag(λ) Qᵀ` with λ falling slowly from 1.
fn clustered(n: usize, seed: u64) -> (DMatrix<f64>, Vec<f64>) {
    let q = DMatrix::<f64>::rnorm_seeded(n, n, seed).qr().q();
    let lambda: Vec<f64> = (0..n).map(|i| 0.99f64.powi(i as i32)).collect();
    (
        &q * DMatrix::from_diagonal(&DVector::from_vec(lambda.clone())) * q.transpose(),
        lambda,
    )
}

#[test]
fn rsvd_with_more_iterations_and_oversampling_separates_a_clustered_spectrum() {
    let (x, lambda) = clustered(300, 3);
    let worst = |s: &DVector<f64>| {
        (0..15)
            .map(|i| (s[i] - lambda[i]).abs())
            .fold(0.0f64, f64::max)
    };
    let (_, s_default, _) = x.rsvd(15).unwrap();
    let args = RsvdArgs {
        power_iters: 20,
        oversample: 10,
    };
    let (_, s_more, _) = x.rsvd_with(15, &args).unwrap();
    assert!(
        worst(&s_more) < 1e-3,
        "20 iterations, 10 oversample: worst error {}",
        worst(&s_more)
    );
    assert!(
        worst(&s_more) < worst(&s_default) / 5.0,
        "more iterations and oversampling should be clearly better: {} vs default {}",
        worst(&s_more),
        worst(&s_default)
    );
}

#[test]
fn rsvd_defaults_are_the_long_standing_settings() {
    // Every existing `rsvd` caller depends on these staying put.
    assert_eq!(
        RsvdArgs::default(),
        RsvdArgs {
            power_iters: 5,
            oversample: 5
        }
    );
}

#[test]
fn rsvd_with_caps_an_oversized_oversample() {
    let x = DMatrix::<f64>::rnorm_seeded(40, 30, 5);
    let args = RsvdArgs {
        power_iters: 2,
        oversample: usize::MAX,
    };
    let (_, s, _) = x.rsvd_with(10, &args).unwrap();
    assert_eq!(s.len(), 10);
}

#[test]
fn rsvd_of_an_empty_matrix_is_an_error() {
    assert!(DMatrix::<f64>::zeros(0, 5).rsvd(3).is_err());
}

/// A seeded sparse matrix with empty rows and columns, as a `(row, col,
/// value)` list.
fn sparse_triplets(nr: usize, nc: usize, per_row: usize, seed: u64) -> Vec<(usize, usize, f64)> {
    use rand::{RngExt, SeedableRng};
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    let mut out = Vec::new();
    // Every third row stays empty.
    for i in (0..nr).filter(|i| i % 3 != 1) {
        for _ in 0..per_row {
            out.push((i, rng.random_range(0..nc), rng.random_range(-1.0..1.0)));
        }
    }
    out
}

fn coo(nr: usize, nc: usize, t: &[(usize, usize, f64)]) -> nalgebra_sparse::coo::CooMatrix<f64> {
    let mut coo = nalgebra_sparse::coo::CooMatrix::new(nr, nc);
    for &(i, j, v) in t {
        coo.push(i, j, v);
    }
    coo
}

fn assert_close(a: &DMatrix<f64>, b: &DMatrix<f64>) {
    assert_eq!(a.shape(), b.shape());
    let scale = b.amax().max(1.0);
    let err = (a - b).amax();
    assert!(err <= 1e-12 * scale, "max abs difference {err}");
}

/// The parallel products agree with nalgebra-sparse's serial ones, for both
/// storage orders, rectangular shapes, duplicate entries and empty rows, and
/// for a matrix large enough to split across threads.
#[test]
fn parallel_sparse_products_match_the_serial_ones() {
    for (nr, nc, per_row) in [(7, 5, 2), (300, 170, 4), (5000, 3000, 9)] {
        let t = sparse_triplets(nr, nc, per_row, (nr * nc) as u64);
        let csr = CsrMatrix::from(&coo(nr, nc, &t));
        let csc = CscMatrix::from(&coo(nr, nc, &t));
        let b = DMatrix::<f64>::rnorm_seeded(nc, 6, 3);
        let bt = DMatrix::<f64>::rnorm_seeded(nr, 6, 4);
        let want = &csr * &b;
        let want_t = csr.transpose() * &bt;
        for op in [SparseOp::from_csr(&csr), SparseOp::from_csc(&csc)] {
            assert_close(&op.matmul(&b), &want);
            assert_close(&op.transpose_matmul(&bt), &want_t);
            assert_eq!((op.num_rows(), op.num_columns()), (nr, nc));
        }
    }
}

/// The sparse randomised SVD gives the dense one's answer, in either storage
/// order.
#[test]
fn sparse_rsvd_matches_dense_rsvd() {
    let (nr, nc) = (400, 250);
    let t = sparse_triplets(nr, nc, 6, 11);
    let csr = CsrMatrix::from(&coo(nr, nc, &t));
    let csc = CscMatrix::from(&coo(nr, nc, &t));
    let dense = DMatrix::from(&csr);
    let args = RsvdArgs {
        power_iters: 8,
        oversample: 10,
    };
    let (_, s, _) = dense.rsvd_with(5, &args).unwrap();
    for (_, s_sparse, _) in [
        csr.rsvd_with(5, &args).unwrap(),
        csc.rsvd_with(5, &args).unwrap(),
    ] {
        let err = (&s_sparse - &s).amax();
        assert!(err <= 1e-9 * s[0], "{s_sparse} vs {s}");
    }
}
