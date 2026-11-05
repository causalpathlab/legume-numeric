// Formulas and constants of `fast_ln` and `fast_ln_gamma` are from fastapprox:
//
// Copyright (C) 2011 Paul Mineiro
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//     * Neither the name of Paul Mineiro nor the names of other contributors may be
//     used to endorse or promote products derived from this software without
//     specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
// IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
// NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

//! Fast, approximate `ln` and `ln Γ` in `f32`: Paul Mineiro's fastapprox.
//!
//! These trade digits for speed. The one thing that must not be lost is the
//! increment `ln Γ(x + y) − ln Γ(x)`, the form a ratio of Dirichlet-multinomials
//! reads, and [`ln_gamma_increment`] keeps it.

/// `ln x` for a positive normal `x`: the float's bits read as its `log₂`, then
/// corrected. About `1e-4`. Any other `x` gets a plausible number, not a `NaN`.
#[inline(always)]
pub fn fast_ln(x: f32) -> f32 {
    let i = x.to_bits();
    let m = f32::from_bits((i & 0x007F_FFFF) | 0x3f00_0000);
    let y = i as i32 as f32 * 1.192_092_9e-7;
    std::f32::consts::LN_2 * (y - 124.225_52 - 1.498_030_3 * m - 1.725_88 / (0.352_088_7 + m))
}

/// `ln Γ(x)` for `x` from about `1e-38` to `7e12`, where the product below stays
/// a normal float:
/// `−2.0810615 − x + 1/(12(x+3)) − ln(x (x+1) (x+2)) + (x + 2.5) ln(x+3)`.
#[inline(always)]
fn fast_ln_gamma(x: f32) -> f32 {
    let xp3 = 3.0 + x;
    -2.081_061_5 - x + 0.083_333_3 / xp3 - fast_ln(x * (1.0 + x) * (2.0 + x))
        + (2.5 + x) * fast_ln(xp3)
}

/// From here up, the slope of `ln Γ` is close enough to `ln(x − ½)` for the
/// midpoint rule. Below, `ln Γ` is small, so its difference keeps a noise of about
/// `1e-4`, whatever `y` is, rather than losing all its digits.
const MIDPOINT_FROM: f32 = 8.0;

/// `ln Γ(x + y) − ln Γ(x)` for `x > 0` and `y` from above zero up to about `1e12`,
/// off by about a percent of `y`, plus the noise above, which can outweigh a `y` of
/// `1e-3` or less.
///
/// Where `x` is well above `y`, the common case, the expansion
/// `y ln x + c₂/x − c₃/x² + c₄/x³` reads only `ln x` and `1/x`, which a caller scoring many
/// `y` against the same `x` can keep in an [`LnTable`].
///
/// When `y` is small against `x`, subtracting two `ln Γ` cancels nearly all of
/// their `f32` digits, and past `x ≈ 2e7` `x + y` no longer even differs from
/// `x`. Short of the expansion there is the midpoint rule: `∫ ln(t − ½) dt` over
/// `[x, x + y]` needs no subtraction, and with `mid = x + (y − 1)/2` and
/// `r = y/mid`, it is `y · (ln(mid) + 1/(24 mid²) − r²/24 − r⁴/320)`, the second term
/// from the slope being `ln(t − ½) + 1/(24 (t − ½)²)`, the others to the order that
/// keeps it close to the difference where the two meet, at `x = y`. When `y` is not
/// below `x`, the difference cancels nothing and is used as is.
#[inline(always)]
pub fn ln_gamma_increment(x: f32, y: f32) -> f32 {
    debug_assert!(x > 0.0, "{x}");
    let inv = 1.0 / x;
    debug_assert!(y >= 0.0, "{y}");
    if inv > expansion_limit(y) {
        short_of_expansion(x, y)
    } else {
        ExpansionCoefficients::of(y).expansion(fast_ln(x), inv)
    }
}

/// [`ln_gamma_increment`] across `run`, into `out`, bit for bit. The expansion over
/// all of it, a few products per cell; then the other forms where it does not hold,
/// so that a run needing none pays no branch per cell.
#[inline(always)]
pub fn ln_gamma_increments(run: LnRun<'_>, y: f32, out: &mut [f32]) {
    assert_eq!(run.xs.len(), out.len());
    debug_assert!(y >= 0.0, "{y}");
    let coefficients = ExpansionCoefficients::of(y);
    let limit = expansion_limit(y);
    let mut any_fails = false;
    for ((inc, &ln_x), &inv) in out.iter_mut().zip(run.ln_xs).zip(run.inv_xs) {
        *inc = coefficients.expansion(ln_x, inv);
        any_fails |= inv > limit;
    }
    if any_fails {
        for ((inc, &x), &inv) in out.iter_mut().zip(run.xs).zip(run.inv_xs) {
            if inv > limit {
                *inc = short_of_expansion(x, y);
            }
        }
    }
}

/// From here up in `x`, against a `y` of at most one, the expansion holds.
const EXPANSION_FROM: f32 = 8.0;

/// Past this `y`, `c₄` is near the top of what `f32` holds.
const EXPANSION_UP_TO: f32 = 1e9;

/// The largest `1/x` at which the expansion holds: `x` at least `8 max(1, y)`. Its
/// next term is about `y (y/x)⁴ / 20`, about `1e-5` of `y` there, under what
/// `fast_ln` itself resolves. No `x` qualifies for a `y` past
/// [`EXPANSION_UP_TO`], not even an infinite one.
#[inline(always)]
fn expansion_limit(y: f32) -> f32 {
    if y > EXPANSION_UP_TO {
        -1.0
    } else {
        1.0 / (EXPANSION_FROM * y.max(1.0))
    }
}

/// `y`, `c₂ = y (y − 1) / 2`, `c₃ = c₂ (2y − 1) / 6` and `c₄ = c₂² / 3`.
struct ExpansionCoefficients {
    y: f32,
    c2: f32,
    c3: f32,
    c4: f32,
}

impl ExpansionCoefficients {
    #[inline(always)]
    fn of(y: f32) -> Self {
        let c2 = 0.5 * y * (y - 1.0);
        // Reciprocals, as a division by these constants is not folded and costs a pass.
        ExpansionCoefficients {
            y,
            c2,
            c3: c2 * (2.0 * y - 1.0) * (1.0 / 6.0),
            c4: c2 * c2 * (1.0 / 3.0),
        }
    }

    #[inline(always)]
    fn expansion(&self, ln_x: f32, inv: f32) -> f32 {
        self.y * ln_x + inv * (self.c2 - inv * (self.c3 - inv * self.c4))
    }
}

/// The midpoint form or the difference, for where the expansion does not hold.
#[inline(always)]
fn short_of_expansion(x: f32, y: f32) -> f32 {
    if y < x && x >= MIDPOINT_FROM {
        let mid = x + 0.5 * (y - 1.0);
        let inv_mid = 1.0 / mid;
        let ratio = y * inv_mid;
        let squared = ratio * ratio;
        let slope = fast_ln(mid) + inv_mid * inv_mid * (1.0 / 24.0);
        y * (slope - squared * (1.0 / 24.0 + squared * (1.0 / 320.0)))
    } else {
        fast_ln_gamma(x + y) - fast_ln_gamma(x)
    }
}

/// A list of `x` with what [`ln_gamma_increments`] reads of each, kept in step.
#[derive(Clone)]
pub struct LnTable {
    xs: Vec<f32>,
    ln_xs: Vec<f32>,
    inv_xs: Vec<f32>,
}

/// A stretch of an [`LnTable`], its slices of one length.
#[derive(Copy, Clone)]
pub struct LnRun<'a> {
    xs: &'a [f32],
    ln_xs: &'a [f32],
    inv_xs: &'a [f32],
}

impl LnTable {
    /// A table of `n` cells, each to be [`set`](Self::set) before it is read.
    pub fn new(n: usize) -> Self {
        LnTable {
            xs: vec![f32::NAN; n],
            ln_xs: vec![f32::NAN; n],
            inv_xs: vec![f32::INFINITY; n],
        }
    }

    /// Set cell `i` to `x`, with what the increments read of it.
    #[inline]
    pub fn set(&mut self, i: usize, x: f32) {
        debug_assert!(x > 0.0, "{x}");
        self.xs[i] = x;
        self.ln_xs[i] = fast_ln(x);
        self.inv_xs[i] = 1.0 / x;
    }

    /// The cells of `range`, to pass to [`ln_gamma_increments`].
    #[inline]
    pub fn run(&self, range: std::ops::Range<usize>) -> LnRun<'_> {
        LnRun {
            xs: &self.xs[range.clone()],
            ln_xs: &self.ln_xs[range.clone()],
            inv_xs: &self.inv_xs[range],
        }
    }

    /// Cell `i`: its `x`, `ln x` and `1/x`.
    pub fn cell(&self, i: usize) -> [f32; 3] {
        [self.xs[i], self.ln_xs[i], self.inv_xs[i]]
    }
}

#[cfg(test)]
#[path = "fast_math_tests.rs"]
mod tests;
