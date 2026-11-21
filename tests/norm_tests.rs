use approx::assert_abs_diff_eq;
use legume_numeric::matrix::traits::{MatOps, SampleOps};

#[test]
fn dmatrix_test() {
    let mut xx = nalgebra::DMatrix::<f32>::runif(100, 10);
    xx.normalize_columns_inplace();

    for j in 0..xx.ncols() {
        let norm = xx.column(j).norm();
        // Re-norming a normalized column sums 100 f32 squares and takes a
        // square root, so a couple of ULP of drift is arithmetic, not a bug.
        // The default epsilon is exactly `f32::EPSILON`, which this trips on
        // roughly one seed in ten.
        assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-6);
    }
}

#[test]
fn winsorize_clips_only_the_extreme_tails() {
    use legume_numeric::matrix::dmatrix_util::winsorize_rows_inplace;
    // Row 0: 0..9 and one far outlier; row 1 constant (MAD 0, untouched).
    let mut row0: Vec<f32> = (0..10).map(|v| v as f32).collect();
    row0.push(1000.0);
    let mut m = nalgebra::DMatrix::<f32>::from_fn(2, 11, |r, c| if r == 0 { row0[c] } else { 3.0 });
    let before = m.clone();
    assert_eq!(winsorize_rows_inplace(&mut m, 0.0), 0);
    assert_eq!(m, before);
    assert_eq!(winsorize_rows_inplace(&mut m, 2.0), 1);
    // Median 5, MAD 3 · 1.4826, so the ceiling is 5 + 2 · 4.4478.
    assert_abs_diff_eq!(m[(0, 10)], 5.0 + 2.0 * 3.0 * 1.4826, epsilon = 1e-4);
    for c in 0..10 {
        assert_eq!(m[(0, c)], c as f32);
        assert_eq!(m[(1, c)], 3.0);
    }
}
