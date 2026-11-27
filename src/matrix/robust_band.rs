//! Robust (median ± MAD) outlier bands, for cell QC and any per-unit metric.
//!
//! Moved from data-beans' `qc_lib` (which re-exports it) so crates below
//! data-beans can use it without depending on data-beans.

/// Which side(s) of the robust band count as outliers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tail {
    /// Flag values far *below* the median (e.g. low counts / few genes).
    Lower,
    /// Flag values far *above* the median (e.g. high mito fraction).
    Upper,
    /// Flag both extremes.
    Both,
}

/// Median of a non-empty slice via `O(n)` quickselect (reorders in place).
fn median_in_place(xs: &mut [f32]) -> f32 {
    let mid = xs.len() / 2;
    xs.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    xs[mid]
}

/// The robust band [`robust_outlier_keep`] tests against: `median ± n_mads ·
/// MAD · 1.4826`, on the `log1p` scale when asked.
#[derive(Clone, Copy, Debug)]
pub struct RobustBand {
    pub median: f32,
    pub lo: f32,
    pub hi: f32,
}

impl RobustBand {
    /// The band on the values' own scale, undoing `log1p`.
    pub fn raw(self, log1p: bool) -> Self {
        let back = |v: f32| if log1p { v.exp_m1() } else { v };
        Self {
            median: back(self.median),
            lo: back(self.lo),
            hi: back(self.hi),
        }
    }
}

/// A value on the scale its band is fitted on: `ln(1 + v)` for counts.
fn band_scale(v: f32, log1p: bool) -> f32 {
    if log1p {
        v.max(0.0).ln_1p()
    } else {
        v
    }
}

/// [`RobustBand`] of `values` (the `consider`ed ones, when given); `None`
/// when none is finite, or when they have no spread (a MAD of 0).
pub fn robust_band(
    values: &[f32],
    n_mads: f32,
    log1p: bool,
    consider: Option<&[bool]>,
) -> Option<RobustBand> {
    let xform = |v: f32| band_scale(v, log1p);

    // Band statistics computed over the considered subset only.
    let mut xs: Vec<f32> = values
        .iter()
        .enumerate()
        .filter(|(i, _)| consider.is_none_or(|c| c[*i]))
        .map(|(_, &v)| xform(v))
        .filter(|v| v.is_finite())
        .collect();
    if xs.is_empty() {
        return None;
    }
    // O(n) median via quickselect (no full sort).
    let median = median_in_place(&mut xs);

    let mut dev: Vec<f32> = xs.iter().map(|&v| (v - median).abs()).collect();
    let mad = median_in_place(&mut dev) * 1.4826;
    // No spread: more than half the values are one value (a mito fraction
    // of 0 in most nuclei, say). A band of width 0 would call every other
    // value an outlier, so there is none.
    if mad <= 0.0 {
        return None;
    }

    Some(RobustBand {
        median,
        lo: median - n_mads * mad,
        hi: median + n_mads * mad,
    })
}

/// A robust standardisation of one column of values: `(v − median) / scale`,
/// softly clipped to `±clip` ([`RobustScale::apply`]), so a long tail cannot stretch the rest of the column
/// or take a cluster of its own. `scale` is the larger of `1.4826 · MAD` and
/// `IQR / 1.349`, both normal-consistent and blind to a thin tail; the IQR
/// keeps two clumps of like size apart, where the MAD alone measures only
/// the clump the median falls in. When both are 0 (most values one value),
/// it is `1.2533 ·` the mean absolute deviation from the median.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RobustScale {
    pub median: f32,
    pub scale: f32,
    pub clip: f32,
}

impl RobustScale {
    /// The standardisation of `values`, its finite ones; `None` when none
    /// is finite or they do not vary.
    #[must_use]
    pub fn fit(values: &[f32], clip: f32) -> Option<Self> {
        let mut xs: Vec<f32> = values.iter().copied().filter(|v| v.is_finite()).collect();
        if xs.is_empty() {
            return None;
        }
        let median = median_in_place(&mut xs);
        let q = crate::matrix::utils::quantiles(&xs, &[0.25, 0.75]);
        let iqr = (q[1] - q[0]) / 1.349;
        let mut dev: Vec<f32> = xs.iter().map(|&v| (v - median).abs()).collect();
        let mean_dev = dev.iter().map(|&v| f64::from(v)).sum::<f64>() / dev.len() as f64;
        let mad = 1.4826 * median_in_place(&mut dev);
        let scale = if mad.max(iqr) > 0.0 {
            mad.max(iqr)
        } else {
            (1.2533 * mean_dev) as f32
        };
        (scale > 0.0).then_some(Self {
            median,
            scale,
            clip,
        })
    }

    /// `v` standardised and softly clipped, `clip · tanh(x / clip)`: about `x`
    /// within a few SDs, never past `±clip`, and a tail kept in order rather
    /// than piled on the bound as a hard clip would (only values beyond about
    /// `9 · clip` meet at it, where `tanh` rounds to 1). A non-finite `v` is 0.
    #[must_use]
    pub fn apply(&self, v: f32) -> f32 {
        if v.is_finite() {
            let x = (v - self.median) / self.scale;
            self.clip * (x / self.clip).tanh()
        } else {
            0.0
        }
    }
}

/// Each column of `x` standardised in place by its [`RobustScale`]; a column
/// that does not vary is set to 0 and has `None`.
pub fn robust_scale_columns_inplace(
    x: &mut nalgebra::DMatrix<f32>,
    clip: f32,
) -> Vec<Option<RobustScale>> {
    x.column_iter_mut()
        .map(|mut col| {
            let scale = RobustScale::fit(col.as_slice(), clip);
            match scale {
                Some(s) => col.apply(|v| *v = s.apply(*v)),
                None => col.fill(0.0),
            }
            scale
        })
        .collect()
}

/// Robust outlier keep-mask: `keep[i] = false` when `values[i]` falls
/// outside `median ± n_mads · MAD · 1.4826` on the requested `tail`.
/// Count-like metrics should pass `log1p = true` so the band is symmetric
/// on the multiplicative scale. Consolidates the median+MAD idiom used in
/// `cnv::hmm`.
/// `consider` (when `Some`) restricts the median/MAD band to the cells it
/// marks `true` — so e.g. near-empty cells don't contaminate the robust
/// center — while the keep decision is still returned for every cell.
pub fn robust_outlier_keep(
    values: &[f32],
    n_mads: f32,
    tail: Tail,
    log1p: bool,
    consider: Option<&[bool]>,
) -> Vec<bool> {
    let n = values.len();
    let Some(band) = robust_band(values, n_mads, log1p, consider) else {
        return vec![true; n];
    };
    let xform = |v: f32| band_scale(v, log1p);
    let (lo, hi) = (band.lo, band.hi);
    values
        .iter()
        .map(|&raw| {
            let v = xform(raw);
            if !v.is_finite() {
                return false;
            }
            match tail {
                Tail::Lower => v >= lo,
                Tail::Upper => v <= hi,
                Tail::Both => v >= lo && v <= hi,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_robust_scale_ignores_a_long_tail_and_falls_back_when_most_values_tie() {
        // 99 values about 0 with spread 1, one at -100.
        let mut v: Vec<f32> = (0..99).map(|i| (i as f32 - 49.0) / 49.0 * 1.7).collect();
        v.push(-100.0);
        let s = RobustScale::fit(&v, 5.0).unwrap();
        assert!(s.median.abs() < 0.1);
        assert!(s.scale > 1.0 && s.scale < 2.0, "{s:?}");
        assert!(s.apply(-100.0) >= -5.0 && s.apply(-100.0) < -4.99);
        // Tail values a few SDs apart stay apart.
        assert!(s.apply(s.median - 6.0 * s.scale) < s.apply(s.median - 5.0 * s.scale));
        assert!((s.apply(s.median + s.scale) - 1.0).abs() < 0.02);
        // Mostly zeros (a MAD of 0): the mean absolute deviation scales.
        let z: Vec<f32> = (0..100).map(|i| if i < 80 { 0.0 } else { 2.0 }).collect();
        let s = RobustScale::fit(&z, 5.0).unwrap();
        assert_eq!(s.median, 0.0);
        assert!((s.scale - 1.2533 * 0.4).abs() < 1e-4);
        assert!(RobustScale::fit(&[3.0; 10], 5.0).is_none());
        let mut m = nalgebra::DMatrix::from_fn(100, 2, |i, j| if j == 0 { v[i] } else { 1.0 });
        let fits = robust_scale_columns_inplace(&mut m, 5.0);
        assert!(fits[0].is_some() && fits[1].is_none());
        assert!(m[(99, 0)] < -4.99);
        assert!(m.column(1).iter().all(|&x| x == 0.0));
    }

    #[test]
    fn robust_lower_flags_low_outlier() {
        let v = vec![100.0, 110.0, 90.0, 105.0, 95.0, 1.0];
        let keep = robust_outlier_keep(&v, 3.0, Tail::Lower, true, None);
        assert!(!keep[5], "the value 1.0 should be a lower outlier");
        assert!(keep[..5].iter().all(|&k| k), "the bulk should be kept");
        // upper tail must not flag a low value
        let keep_up = robust_outlier_keep(&v, 3.0, Tail::Upper, true, None);
        assert!(keep_up[5], "lower outlier kept under Tail::Upper");
    }

    #[test]
    fn no_spread_flags_no_one() {
        // Most cells read no mito at all: the MAD is 0, and a cell with one
        // mito read is no outlier for it.
        let mut v = vec![0.0f32; 60];
        v.extend([0.01, 0.02, 0.05, 0.3]);
        assert!(robust_band(&v, 5.0, false, None).is_none());
        let keep = robust_outlier_keep(&v, 5.0, Tail::Upper, false, None);
        assert!(keep.iter().all(|&k| k));
    }

    #[test]
    fn robust_uniform_keeps_all() {
        let v = vec![7.0; 20];
        let keep = robust_outlier_keep(&v, 5.0, Tail::Both, true, None);
        assert!(keep.iter().all(|&k| k));
    }

    #[test]
    fn robust_consider_excludes_contaminants_from_band() {
        // A real cluster around 100 plus a large block of near-empty 0s that
        // would drag a naive median/MAD down. With `consider` masking the
        // zeros out of the band, a genuine low real cell (40) is still flagged.
        let mut v = vec![0.0; 12];
        v.extend([100.0, 102.0, 98.0, 101.0, 99.0, 40.0]);
        let mut consider = vec![false; 12];
        consider.extend([true; 6]);
        let keep = robust_outlier_keep(&v, 2.0, Tail::Lower, true, Some(&consider));
        assert!(
            !keep[17],
            "40 is a lower outlier of the real cluster (~100)"
        );
        // Without `consider`, the zeros dominate the median and 40 survives.
        let keep_naive = robust_outlier_keep(&v, 2.0, Tail::Lower, true, None);
        assert!(
            keep_naive[17],
            "naive band (contaminated by zeros) keeps 40"
        );
    }
}
