use super::{fast_ln, fast_ln_gamma, ln_gamma_increment, ln_gamma_increments, LnTable};
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
/// has to scale with `y`, and stay small across the splits between the forms.
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

/// How far crossing from `below` to `above` moves an increment beyond its true slope.
fn jump_error(below: f64, above: f64, y: f64) -> f64 {
    let approx = |x: f64| f64::from(ln_gamma_increment(x as f32, y as f32));
    let exact = |x: f64| Gamma::ln_gamma(x + y).0 - Gamma::ln_gamma(x).0;
    ((approx(above) - approx(below)) - (exact(above) - exact(below))).abs()
}

/// The two forms meet at `x = y`, a common place for `x` to sit when it mostly
/// holds one count like `y`, so crossing it may not shift a score by more than its true
/// slope and a small share of `y`.
#[test]
fn increments_barely_jump_where_the_forms_meet() {
    for y in [1e2f64, 1e3, 1e4, 1e5, 1e6] {
        let error = jump_error(y * 0.999, y * 1.001, y);
        assert!(error <= 2e-4 * y, "y = {y}: a jump off by {error}");
    }
}

/// The cached form gives way to the others at `x = 8 max(1, y)`. The two agreeing
/// there is what keeps crossing it from shifting a score: this far from `x = y` the
/// two sides of the seam have no reason to differ by more than their slope.
#[test]
fn increments_barely_jump_where_the_cached_form_ends() {
    for y in [
        1e-1f64, 0.5, 1.0, 1.05, 1.2, 1.35, 2.0, 3.0, 5.0, 1e1, 1e2, 1e3, 1e4, 1e5,
    ] {
        let seam = 8.0 * y.max(1.0);
        let error = jump_error(seam * 0.999, seam * 1.001, y);
        assert!(error <= 2e-4 * y + 2e-4, "y = {y}: a jump off by {error}");
    }
}

/// Near where the forms hand over, the error is a one-sided bias if the expansion
/// runs too close to `y`, and a score sums a bias where it averages noise out.
/// The absolute term is the noise of `ln Γ` near `x = 8` that `MIDPOINT_FROM` notes.
#[test]
fn increments_stay_within_what_fast_ln_resolves_around_the_seam() {
    for y in [1.0f64, 1.2, 2.0, 4.0, 8.0, 1e1, 1e2, 1e3, 1e4] {
        for x in grid(4.0 * y.max(2.0), 64.0 * y, 1.03) {
            let want = Gamma::ln_gamma(x + y).0 - Gamma::ln_gamma(x).0;
            let got = f64::from(ln_gamma_increment(x as f32, y as f32));
            assert!(
                (got - want).abs() <= 1.6e-4 * y + 1e-4,
                "ln Γ({x} + {y}) − ln Γ({x}): {got} against {want}, {:.2e} of y",
                (got - want).abs() / y
            );
        }
    }
}

/// A pair of totals multiplied makes a `y` of this size, and an overflow in the
/// coefficients would silently wreck a fit, as a sampler reads `inf` as no opinion.
#[test]
fn increments_survive_very_large_counts() {
    for y in [1e8f64, 1e9, 1e10, 1e11, 1e12] {
        for x in grid(y, 1e4 * y, 3.1) {
            let want = Gamma::ln_gamma(x + y).0 - Gamma::ln_gamma(x).0;
            let got = f64::from(ln_gamma_increment(x as f32, y as f32));
            assert!(
                (got - want).abs() <= 1e-2 * y,
                "ln Γ({x} + {y}) − ln Γ({x}): {got} against {want}"
            );
        }
    }
}

/// A value computed through the run and one computed alone must not differ, or
/// rounding alone could tell two callers' answers apart. Up to the largest `y`, whose
/// overflowed coefficients the run must not let through, and at any offset.
#[test]
fn a_run_of_increments_is_the_scalar_bit_for_bit() {
    let xs: Vec<f32> = grid(1e-3, 1e13, 1.7).iter().map(|&x| x as f32).collect();
    let mut table = LnTable::new(xs.len());
    for (i, &x) in xs.iter().enumerate() {
        table.set(i, x);
    }
    let mut out = vec![0.0; xs.len()];
    for y in grid(1e-3, 1e12, 3.3) {
        let y = y as f32;
        for start in [0, 7] {
            let out = &mut out[start..];
            ln_gamma_increments(table.run(start..xs.len()), y, out);
            for (&x, &got) in xs[start..].iter().zip(out.iter()) {
                assert_eq!(
                    got.to_bits(),
                    ln_gamma_increment(x, y).to_bits(),
                    "{x}, {y}"
                );
            }
        }
    }
}

/// A cell left unset must not pass for a plausible increment.
#[test]
fn a_cell_never_set_gives_no_plausible_increment() {
    let table = LnTable::new(1);
    let mut out = [0.0];
    ln_gamma_increments(table.run(0..1), 2.0, &mut out);
    assert!(out[0].is_nan(), "{}", out[0]);
}
