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

/// `ln Γ(x + y) − ln Γ(x)` for `x, y > 0`, off by about a percent of `y`, plus the
/// noise above, which can outweigh a `y` of `1e-3` or less.
///
/// When `y` is small against `x`, subtracting two `ln Γ` cancels nearly all of
/// their `f32` digits, and past `x ≈ 2e7` `x + y` no longer even differs from
/// `x`. There `∫ ln(t − ½) dt` over `[x, x + y]` needs no subtraction: with
/// `mid = x + (y − 1)/2` and `r = y/mid`, it is `y · (ln(mid) − r²/24 − r⁴/320)`
/// to the order that keeps the two forms close where they meet, at `x = y`.
/// When `y` is not below `x`, the difference cancels nothing and is used as is.
#[inline(always)]
pub fn ln_gamma_increment(x: f32, y: f32) -> f32 {
    if midpoint_holds(x, y) {
        midpoint_increment(x, y)
    } else {
        difference_increment(x, y)
    }
}

/// [`ln_gamma_increment`] at each of `xs`, into `out`, bit for bit: the midpoint
/// form across the run, which vectorizes, then the other form where it does not
/// hold, so that a run needing none pays no branch per cell. The check rides in the
/// same loop: that loop vectorizes wider than the midpoint form alone, from a run
/// of about 16 up, though it is slower at about 8 to 12.
#[inline(always)]
pub fn ln_gamma_increments(xs: &[f32], y: f32, out: &mut [f32]) {
    debug_assert_eq!(xs.len(), out.len());
    let mut any_fails = false;
    for (inc, &x) in out.iter_mut().zip(xs) {
        *inc = midpoint_increment(x, y);
        any_fails |= !midpoint_holds(x, y);
    }
    if any_fails {
        for (inc, &x) in out.iter_mut().zip(xs) {
            if !midpoint_holds(x, y) {
                *inc = difference_increment(x, y);
            }
        }
    }
}

/// Whether [`ln_gamma_increment`] takes the midpoint form at these arguments.
#[inline(always)]
fn midpoint_holds(x: f32, y: f32) -> bool {
    y < x && x >= MIDPOINT_FROM
}

/// The midpoint form alone: branch-free, so a loop over `x` can vectorize.
#[inline(always)]
fn midpoint_increment(x: f32, y: f32) -> f32 {
    let mid = x + 0.5 * (y - 1.0);
    let ratio = y / mid;
    let squared = ratio * ratio;
    // Reciprocals, as a division by these constants is not folded and costs a pass.
    y * (fast_ln(mid) - squared * (1.0 / 24.0 + squared * (1.0 / 320.0)))
}

/// The difference form alone: for where the midpoint form does not hold.
#[inline(always)]
fn difference_increment(x: f32, y: f32) -> f32 {
    fast_ln_gamma(x + y) - fast_ln_gamma(x)
}

#[cfg(test)]
#[path = "fast_math_tests.rs"]
mod tests;
