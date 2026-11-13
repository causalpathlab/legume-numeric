use super::*;
use crate::matrix::traits::{SampleOps, SymmetricTopEigen, TopEigenArgs};

/// `Q diag(λ) Qᵀ` for a seeded random orthogonal `Q`.
fn with_spectrum(lambda: &[f64], seed: u64) -> DMatrix<f64> {
    let n = lambda.len();
    let q = DMatrix::<f64>::rnorm_seeded(n, n, seed).qr().q();
    let a = &q * DMatrix::from_diagonal(&DVector::from_column_slice(lambda)) * q.transpose();
    (&a + a.transpose()) * 0.5
}

/// A symmetric Gaussian (Wigner) matrix, spectrum roughly `[-2, 2]`.
fn wigner(n: usize, seed: u64) -> DMatrix<f64> {
    let x = DMatrix::<f64>::rnorm_seeded(n, n, seed);
    (&x + x.transpose()) / (2.0 * n as f64).sqrt()
}

/// Wigner noise plus `spikes` planted on random orthonormal directions, and
/// `sinks` planted on others: large negative eigenvalues a ranking by `|λ|`
/// would take for leading ones.
fn planted(n: usize, spikes: &[f64], sinks: &[f64], seed: u64) -> DMatrix<f64> {
    let m = spikes.len() + sinks.len();
    let u = DMatrix::<f64>::rnorm_seeded(n, m, seed + 1).qr().q();
    let d: Vec<f64> = spikes.iter().chain(sinks).copied().collect();
    wigner(n, seed) + &u * DMatrix::from_diagonal(&DVector::from_vec(d)) * u.transpose()
}

/// The reference: all eigenpairs by nalgebra, descending.
fn exact(a: &DMatrix<f64>) -> (DVector<f64>, DMatrix<f64>) {
    sorted_eigen(a.clone())
}

/// Values within `tol` relative to the spectral scale, and each vector
/// within `vec_tol` of the reference (up to sign).
fn assert_agrees(a: &DMatrix<f64>, rank: usize, got: &TopEigen<DMatrix<f64>, DVector<f64>>) {
    let (values, vectors) = exact(a);
    let scale = values[0].abs().max(values[values.len() - 1].abs());
    assert_eq!(got.values.len(), rank);
    assert_eq!(got.vectors.shape(), (a.nrows(), rank));
    assert!(got.converged, "not converged in {} iterations", got.iters);
    for i in 0..rank {
        let err = (got.values[i] - values[i]).abs();
        assert!(
            err <= 1e-8 * scale,
            "value {i}: {} vs exact {} (err {err})",
            got.values[i],
            values[i]
        );
    }
    // Vectors are well defined where the gap is; the planted spikes are
    // separated, so compare each.
    for i in 0..rank {
        let gap = if i + 1 < values.len() {
            values[i] - values[i + 1]
        } else {
            f64::INFINITY
        };
        let gap = if i > 0 {
            gap.min(values[i - 1] - values[i])
        } else {
            gap
        };
        if gap > 1e-3 * scale {
            let cos = got.vectors.column(i).dot(&vectors.column(i)).abs();
            assert!(cos >= 1.0 - 1e-8, "vector {i}: cosine {cos}");
        }
    }
    let gram = got.vectors.tr_mul(&got.vectors);
    let off = (&gram - DMatrix::identity(rank, rank)).amax();
    assert!(off <= 1e-10, "not orthonormal: {off}");
}

#[test]
fn matches_symmetric_eigen_with_planted_spikes() {
    let spikes: Vec<f64> = (0..8).map(|i| 20.0 - 2.0 * i as f64).collect();
    let a = planted(400, &spikes, &[], 1);
    let got = a.top_eigen(8).unwrap();
    assert_agrees(&a, 8, &got);
}

#[test]
fn matches_symmetric_eigen_into_the_bulk() {
    // Past the spikes into the noise bulk, where the spectrum is dense.
    let a = planted(300, &[9.0, 7.0, 5.0], &[], 2);
    let got = a.top_eigen(12).unwrap();
    assert_agrees(&a, 12, &got);
}

#[test]
fn large_negative_eigenvalues_are_not_leading() {
    // The sinks dwarf the spikes in magnitude; RSVD would return them.
    let a = planted(400, &[10.0, 8.0, 6.0, 4.0], &[-60.0, -50.0, -40.0], 3);
    let got = a.top_eigen(4).unwrap();
    assert_agrees(&a, 4, &got);
    assert!(got.values.iter().all(|&v| v > 3.0), "{}", got.values);

    // A spectrum that is mostly negative, top values small and positive.
    let mut lambda: Vec<f64> = (0..250).map(|i| -100.0 + 0.3 * i as f64).collect();
    lambda.extend([0.5, 1.0, 1.5, 2.0, 2.5]);
    let a = with_spectrum(&lambda, 4);
    let got = a.top_eigen(5).unwrap();
    assert_agrees(&a, 5, &got);
    let want = [2.5, 2.0, 1.5, 1.0, 0.5];
    for (g, w) in got.values.iter().zip(want) {
        assert!((g - w).abs() < 1e-6, "{} vs {w}", got.values);
    }
}

#[test]
fn all_negative_spectrum() {
    let lambda: Vec<f64> = (0..200).map(|i| -1.0 - i as f64).collect();
    let a = with_spectrum(&lambda, 5);
    let got = a.top_eigen(6).unwrap();
    assert_agrees(&a, 6, &got);
    assert!((got.values[0] + 1.0).abs() < 1e-8);
}

#[test]
fn identity_like_spectrum_converges() {
    // Lanczos breaks down at once on a multiple of the identity.
    let a = DMatrix::<f64>::identity(100, 100) * 3.0;
    let got = a.top_eigen(5).unwrap();
    assert!(got.converged);
    assert!(got.values.iter().all(|&v| (v - 3.0).abs() < 1e-12));
}

#[test]
fn warm_start_converges_in_fewer_iterations() {
    // A slowly decaying spectrum with a deep negative end, so a cold start
    // needs several cycles.
    let lambda: Vec<f64> = (0..500_i32)
        .map(|i| {
            if i < 5 {
                -20.0 + i as f64
            } else {
                0.98f64.powi(i)
            }
        })
        .collect();
    let a = with_spectrum(&lambda, 6);
    let first = a.top_eigen(20).unwrap();
    assert!(first.converged);

    // Refit on a nearby matrix, as principal-axis factoring does after
    // updating the diagonal.
    let shift = DVector::<f64>::from_fn(500, |i, _| 0.02 * ((i % 7) as f64 - 3.0));
    let b = &a + DMatrix::from_diagonal(&shift);
    let args = TopEigenArgs::default();
    let cold = b.top_eigen_with(20, &args, None).unwrap();
    let warm = b.top_eigen_with(20, &args, Some(&first.vectors)).unwrap();
    assert_agrees(&b, 20, &warm);
    assert!(
        warm.iters < cold.iters,
        "warm {} vs cold {} iterations",
        warm.iters,
        cold.iters
    );

    // Starting from the answer itself needs no iteration.
    let again = a.top_eigen_with(20, &args, Some(&first.vectors)).unwrap();
    assert_eq!(again.iters, 0);
}

#[test]
fn warm_start_of_the_wrong_height_is_an_error() {
    let a = wigner(50, 7);
    let w = DMatrix::<f64>::zeros(49, 3);
    assert!(a
        .top_eigen_with(3, &TopEigenArgs::default(), Some(&w))
        .is_err());
}

#[test]
fn rank_zero_and_rank_at_least_n() {
    let a = planted(60, &[5.0], &[-9.0], 8);
    let none = a.top_eigen(0).unwrap();
    assert_eq!(none.values.len(), 0);
    assert_eq!(none.vectors.shape(), (60, 0));

    let (values, _) = exact(&a);
    for rank in [60, 61, 1000] {
        let all = a.top_eigen(rank).unwrap();
        assert_eq!(all.values.len(), 60);
        assert_eq!(all.vectors.shape(), (60, 60));
        assert!((&all.values - &values).amax() < 1e-10);
        assert!(all.values[59] < -8.0, "negative value kept last");
    }
}

#[test]
fn non_square_is_an_error() {
    assert!(DMatrix::<f64>::zeros(4, 5).top_eigen(2).is_err());
}

#[test]
fn non_finite_entries_are_an_error() {
    let mut a = wigner(80, 12);
    a[(3, 5)] = f64::NAN;
    a[(5, 3)] = f64::NAN;
    assert!(a.top_eigen(4).is_err());
}

#[test]
fn hollow_matrix_with_zero_rayleigh_quotients() {
    // The adjacency of a bipartite graph: ±1 spectrum pairs and a zero
    // diagonal, so the Lanczos α are all near zero.
    let (p, q) = (60, 90);
    let b = DMatrix::<f64>::rnorm_seeded(p, q, 13);
    let mut a = DMatrix::<f64>::zeros(p + q, p + q);
    a.view_mut((0, p), (p, q)).copy_from(&b);
    a.view_mut((p, 0), (q, p)).copy_from(&b.transpose());
    let got = a.top_eigen(5).unwrap();
    assert_agrees(&a, 5, &got);
}

#[test]
fn f32_matches_f64() {
    let a = planted(300, &[12.0, 9.0, 7.0, 5.0], &[-30.0], 9);
    let want = a.top_eigen(4).unwrap();
    let a32 = a.map(|x| x as f32);
    let args = TopEigenArgs {
        tol: 1e-4,
        ..TopEigenArgs::default()
    };
    let got = a32.top_eigen_with(4, &args, None).unwrap();
    assert!(got.converged);
    for i in 0..4 {
        assert!((got.values[i] as f64 - want.values[i]).abs() < 1e-3);
        let cos: f64 = (0..300)
            .map(|j| got.vectors[(j, i)] as f64 * want.vectors[(j, i)])
            .sum();
        assert!(cos > 1.0 - 1e-4, "vector {i}: cosine {cos}");
    }
}

#[test]
fn parallel_product_matches_serial() {
    for n in [5, 70, 333] {
        let a = wigner(n, n as u64);
        let x = DMatrix::<f64>::rnorm_seeded(n, 7, 3);
        let err = (sym_times(&a, &x) - &a * &x).amax();
        assert!(err < 1e-12, "n = {n}: {err}");
    }
}

#[cfg(feature = "tensor")]
#[test]
fn tensor_matches_dmatrix() {
    use candle_core::{Device, Tensor};
    let a = planted(300, &[10.0, 8.0, 6.0, 5.0, 4.0], &[-25.0], 10);
    let want = a.top_eigen(5).unwrap();
    let t = a.to_tensor(&Device::Cpu).unwrap();
    let got = t.top_eigen(5).unwrap();
    assert!(got.converged);
    let values: Vec<f64> = got.values.to_vec1().unwrap();
    let vectors = DMatrix::<f64>::from_tensor(&got.vectors).unwrap();
    for i in 0..5 {
        assert!((values[i] - want.values[i]).abs() < 1e-9, "{values:?}");
        let cos = vectors.column(i).dot(&want.vectors.column(i));
        assert!(cos > 1.0 - 1e-9, "vector {i}: cosine {cos}");
    }

    // f32 tensors, a warm start, and both edge cases.
    let t32 = t.to_dtype(candle_core::DType::F32).unwrap();
    let args = TopEigenArgs {
        tol: 1e-4,
        ..TopEigenArgs::default()
    };
    let warm = t32.top_eigen_with(5, &args, Some(&got.vectors)).unwrap();
    assert!(warm.converged);
    assert_eq!(warm.values.dtype(), candle_core::DType::F32);
    assert_eq!(t.top_eigen(0).unwrap().vectors.dims(), &[300, 0]);
    assert_eq!(t.top_eigen(400).unwrap().values.dims(), &[300]);
    assert!(Tensor::zeros((3, 4), candle_core::DType::F64, &Device::Cpu)
        .unwrap()
        .top_eigen(1)
        .is_err());
}

#[cfg(feature = "ndarray")]
#[test]
fn ndarray_matches_dmatrix() {
    use ndarray::Array2;
    let a = planted(300, &[10.0, 8.0, 6.0, 5.0, 4.0], &[-25.0], 10);
    let want = a.top_eigen(5).unwrap();
    let arr = Array2::from_shape_fn((300, 300), |(i, j)| a[(i, j)]);
    let got = arr.top_eigen(5).unwrap();
    assert!(got.converged);
    assert_eq!(got.vectors.dim(), (300, 5));
    for i in 0..5 {
        assert!((got.values[i] - want.values[i]).abs() < 1e-9);
        let cos: f64 = (0..300)
            .map(|j| got.vectors[(j, i)] * want.vectors[(j, i)])
            .sum();
        assert!(cos > 1.0 - 1e-9, "vector {i}: cosine {cos}");
    }

    // f32, a warm start, the edge cases, and a non-square input.
    let arr32 = arr.mapv(|x| x as f32);
    let args = TopEigenArgs {
        tol: 1e-4,
        ..TopEigenArgs::default()
    };
    let warm32 = got.vectors.mapv(|x| x as f32);
    let warm = arr32.top_eigen_with(5, &args, Some(&warm32)).unwrap();
    assert!(warm.converged);
    assert!((warm.values[0] as f64 - want.values[0]).abs() < 1e-3);
    assert_eq!(arr.top_eigen(0).unwrap().vectors.dim(), (300, 0));
    assert_eq!(arr.top_eigen(400).unwrap().values.len(), 300);
    assert!(Array2::<f64>::zeros((3, 4)).top_eigen(1).is_err());
}

/// A principal-axis-factoring matrix like data-beans' simulator fits: the
/// sample correlation of a `factors`-factor model with communalities on the
/// diagonal, which leaves it indefinite.
fn reduced_correlation(n: usize, factors: usize, samples: usize, seed: u64) -> DMatrix<f64> {
    let w = DMatrix::<f64>::rnorm_seeded(n, factors, seed) * (0.6 / (factors as f64).sqrt());
    let z = DMatrix::<f64>::rnorm_seeded(factors, samples, seed + 1);
    let e = DMatrix::<f64>::rnorm_seeded(n, samples, seed + 2) * 0.8;
    let mut y = &w * z + e;
    for mut row in y.row_iter_mut() {
        let mean = row.mean();
        row.add_scalar_mut(-mean);
        let sd = row.norm();
        row /= sd;
    }
    let mut r = &y * y.transpose();
    for i in 0..n {
        let comm = (0..n)
            .filter(|&j| j != i)
            .map(|j| r[(i, j)].abs())
            .fold(0.0, f64::max);
        r[(i, i)] = comm;
    }
    r
}

/// Timing note for the data-beans case, `n = 2000, rank = 100`; run with
/// `cargo test --release top_eigen_benchmark -- --ignored --nocapture`.
#[test]
#[ignore]
fn top_eigen_benchmark() {
    let a = reduced_correlation(2000, 100, 1000, 11);
    let t0 = std::time::Instant::now();
    let got = a.top_eigen(100).unwrap();
    let fast = t0.elapsed();
    let warm = a
        .top_eigen_with(100, &TopEigenArgs::default(), Some(&got.vectors))
        .unwrap();
    let t1 = std::time::Instant::now();
    let (values, _) = exact(&a);
    let full = t1.elapsed();
    let err = (0..100)
        .map(|i| (got.values[i] - values[i]).abs())
        .fold(0.0, f64::max);
    println!(
        "top_eigen: {fast:?} ({} iters, converged {}), warm restart {} iters; \
         symmetric_eigen: {full:?}; max value error {err:e}; λ_min {}",
        got.iters, got.converged, warm.iters, values[1999]
    );
    assert!(got.converged);
    assert!(err < 1e-6 * values[0]);
}
