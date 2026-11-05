use super::{fast_ln, fast_ln_gamma, ln_gamma_increment, ln_gamma_increments};
use special::Gamma;

/// Log-spaced grid over `[lo, hi]`.
fn grid(lo: f64, hi: f64, ratio: f64) -> Vec<f64> {
    let mut xs = Vec::new();
    let mut x = lo;
    while x <= hi {
        xs.push(x);
        x *= ratio;
    }
    xs
}

/// These trade digits for speed, so this asks for a percent, not for digits.
fn close(got: f64, want: f64) -> bool {
    (got - want).abs() <= 1e-2 * want.abs() + 0.03
}

#[test]
fn fast_ln_is_near_the_library() {
    for x in grid(1e-30, 1e30, 1.37) {
        let got = f64::from(fast_ln(x as f32));
        assert!(close(got, x.ln()), "ln({x}): {got} against {}", x.ln());
    }
}

#[test]
fn fast_ln_gamma_is_near_the_library() {
    for x in grid(1e-3, 1e12, 1.13) {
        let want = Gamma::ln_gamma(x).0;
        let got = f64::from(fast_ln_gamma(x as f32));
        assert!(close(got, want), "ln Γ({x}): {got} against {want}");
    }
}

/// A ratio of Dirichlet-multinomials is `ln Γ(x + y) − ln Γ(x)` at close arguments, with
/// `ln Γ(x)` of order `x ln x`, so the increment is what has to be near. For
/// whole `y` the truth is `Σ_{i<y} ln(x + i)`, free of that cancellation.
#[test]
fn increments_hold_at_large_counts() {
    for x in grid(1.0, 1e9, 3.1) {
        for y in [1usize, 2, 7, 100, 10_000] {
            let want: f64 = (0..y).map(|i| (x + i as f64).ln()).sum();
            let got = f64::from(ln_gamma_increment(x as f32, y as f32));
            assert!(
                close(got, want),
                "ln Γ({x} + {y}) − ln Γ({x}): {got} against {want}"
            );
        }
    }
}

/// Callers compare differences of increments, so the 1% above is no bound on a
/// decision: at `y = 1000` it would forgive tens of nats. The error
/// has to scale with `y`, and stay small across the split between the two forms.
#[test]
fn the_error_of_an_increment_scales_with_its_count() {
    for x in grid(1e-3, 1e7, 1.31) {
        for y in grid(1e-3, 1e4, 1.9) {
            let want = Gamma::ln_gamma(x + y).0 - Gamma::ln_gamma(x).0;
            let got = f64::from(ln_gamma_increment(x as f32, y as f32));
            // `f64` itself resolves no better than this at large `x`.
            let resolution = 1e-14 * Gamma::ln_gamma(x + y).0.abs();
            assert!(
                (got - want).abs() <= 1e-2 * y + 2e-4 + resolution,
                "ln Γ({x} + {y}) − ln Γ({x}): {got} against {want}"
            );
        }
    }
}

/// Many small counts at small `x` must not add up to a bias:
/// the slope at small `x` is `ψ(x)`, whatever form computes it. The error is fixed
/// by `(x, y)`, so entries that share them add up rather than average out.
#[test]
fn small_counts_do_not_pile_up_a_bias_on_empty_features() {
    for x in grid(0.25, 20.0, 1.01) {
        let y = 0.02;
        let want = Gamma::ln_gamma(x + y).0 - Gamma::ln_gamma(x).0;
        let got = f64::from(ln_gamma_increment(x as f32, y as f32));
        // 5000 such entries in one row may cost under a nat in all.
        assert!(
            5000.0 * (got - want).abs() < 1.0,
            "x = {x}: {got} against {want}"
        );
    }
}

/// The two forms meet at `x = y`, a common place for `x` to sit when it mostly
/// holds one count like `y`, so crossing it may not shift a score by more than its true
/// slope and a small share of `y`.
#[test]
fn increments_barely_jump_where_the_forms_meet() {
    for y in [1e2f64, 1e3, 1e4, 1e5, 1e6] {
        let (below, above) = (y * 0.999, y * 1.001);
        let got = f64::from(ln_gamma_increment(above as f32, y as f32))
            - f64::from(ln_gamma_increment(below as f32, y as f32));
        let want = (Gamma::ln_gamma(above + y).0 - Gamma::ln_gamma(above).0)
            - (Gamma::ln_gamma(below + y).0 - Gamma::ln_gamma(below).0);
        assert!(
            (got - want).abs() <= 2e-4 * y,
            "y = {y}: a jump of {got} against {want}"
        );
    }
}

/// A value computed through the run and one computed alone must not differ, or
/// rounding alone could tell two callers' answers apart.
#[test]
fn a_run_of_increments_is_the_scalar_bit_for_bit() {
    let xs: Vec<f32> = grid(1e-3, 1e7, 1.7).iter().map(|&x| x as f32).collect();
    let mut out = vec![0.0; xs.len()];
    for y in grid(1e-3, 1e5, 3.3) {
        let y = y as f32;
        ln_gamma_increments(&xs, y, &mut out);
        for (&x, &got) in xs.iter().zip(&out) {
            assert_eq!(
                got.to_bits(),
                ln_gamma_increment(x, y).to_bits(),
                "{x}, {y}"
            );
        }
    }
}
