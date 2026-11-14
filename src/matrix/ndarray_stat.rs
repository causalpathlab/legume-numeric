use crate::matrix::{
    common_io::{file_ext, write_lines},
    sparse_stat::{narrow, variance_at},
    traits::{IoOps, RunningStatOps},
};
use ndarray::{stack, ArrayBase, Axis, Data, Dimension, NdIndex, OwnedRepr, RemoveAxis, Zip};

/// Denominator for mean/variance: the number of finite values, or a tiny
/// positive when there are none, so an empty entry gives 0 rather than NaN.
/// (As in `sparse_stat`; adding the pseudo count to every entry would leave
/// a constant entry a spurious sd of about sqrt(1e-8 / n).) The variance is
/// `sparse_stat`'s one-pass `variance_at`, clamped at zero.
#[inline]
fn safe_denom(s0: f64) -> f64 {
    if s0 > 0.0 {
        s0
    } else {
        1e-8
    }
}

/// Adds one value to an entry's count of positives, count of finite
/// values and running sums; a non-finite value only counts as nothing.
#[inline]
fn accumulate(npos: &mut f64, s0: &mut f64, s1: &mut f64, s2: &mut f64, x: f32) {
    if x.is_finite() {
        let v = f64::from(x);
        if v > 0.0 {
            *npos += 1.0;
        }
        *s0 += 1.0;
        *s1 += v;
        *s2 += v * v;
    }
}

/// A container to keep track of sufficient statistics of an arbitrary
/// shape `ndarray`
///
/// Values come in as f32 and results go out as f32, but the counts and sums
/// are kept in f64: an f32 sum stops counting exactly past 2^24, and the
/// one-pass variance loses its digits to cancellation.
///
/// # Type parameters
/// - `S` : The shape of the array
///
#[derive(Clone)]
pub struct RunningStatistics<S>
where
    S: Dimension + RemoveAxis,
{
    npos: ArrayBase<OwnedRepr<f64>, S>,
    s0: ArrayBase<OwnedRepr<f64>, S>,
    s1: ArrayBase<OwnedRepr<f64>, S>,
    s2: ArrayBase<OwnedRepr<f64>, S>,
}

impl<S> RunningStatistics<S>
where
    S: Dimension + RemoveAxis,
{
    /// Create a new RunningStatistics object
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the array
    ///
    /// # Examples
    ///
    /// ```
    /// use legume_numeric::matrix::ndarray_stat::RunningStatistics;
    /// use ndarray::Ix1;
    /// let nrow = 10;
    /// RunningStatistics::new(Ix1(nrow));
    /// ```
    ///
    pub fn new(shape: S) -> Self {
        let npos = ArrayBase::zeros(shape.clone());
        let s0 = ArrayBase::zeros(shape.clone());
        let s1 = ArrayBase::zeros(shape.clone());
        let s2 = ArrayBase::zeros(shape);

        RunningStatistics { npos, s0, s1, s2 }
    }

    pub fn add<V>(&mut self, xx: &ArrayBase<V, S>)
    where
        V: Data<Elem = f32>,
    {
        let xx = xx
            .broadcast(self.s0.raw_dim())
            .expect("incompatible shape for running statistics");
        Zip::from(&mut self.npos)
            .and(&mut self.s0)
            .and(&mut self.s1)
            .and(&mut self.s2)
            .and(&xx)
            .for_each(|npos, s0, s1, s2, &x| accumulate(npos, s0, s1, s2, x));
    }

    pub fn add_element<I>(&mut self, idx: &I, val: f32)
    where
        I: NdIndex<S> + Clone,
    {
        fn get<'a, S, I>(mat: &'a mut ArrayBase<OwnedRepr<f64>, S>, idx: &I) -> &'a mut f64
        where
            S: Dimension + RemoveAxis,
            I: NdIndex<S> + Clone,
        {
            mat.get_mut(idx.clone()).expect("failed to access matrix")
        }

        accumulate(
            get(&mut self.npos, idx),
            get(&mut self.s0, idx),
            get(&mut self.s1, idx),
            get(&mut self.s2, idx),
            val,
        );
    }

    pub fn clear(&mut self) {
        self.npos.fill(0.0);
        self.s0.fill(0.0);
        self.s1.fill(0.0);
        self.s2.fill(0.0);
    }

    /// Frequency of positive values. For a sparse count matrix, this
    /// will reflect the number of non-zero values
    ///
    pub fn count_positives(&self) -> ArrayBase<OwnedRepr<f32>, S> {
        self.npos.mapv(narrow::<f32>)
    }

    /// Sum of values
    pub fn sum(&self) -> ArrayBase<OwnedRepr<f32>, S> {
        self.s1.mapv(narrow::<f32>)
    }

    /// Average statistic
    pub fn mean(&self) -> ArrayBase<OwnedRepr<f32>, S> {
        Zip::from(&self.s0)
            .and(&self.s1)
            .map_collect(|&s0, &s1| narrow(s1 / safe_denom(s0)))
    }

    /// Variance, clamped at zero
    pub fn variance(&self) -> ArrayBase<OwnedRepr<f32>, S> {
        Zip::from(&self.s0)
            .and(&self.s1)
            .and(&self.s2)
            .map_collect(|&s0, &s1, &s2| narrow(variance_at(s1, s2, safe_denom(s0))))
    }

    /// Standard deviation
    pub fn std(&self) -> ArrayBase<OwnedRepr<f32>, S> {
        Zip::from(&self.s0)
            .and(&self.s1)
            .and(&self.s2)
            .map_collect(|&s0, &s1, &s2| narrow(variance_at(s1, s2, safe_denom(s0)).sqrt()))
    }

    pub fn shape(&self) -> &[usize] {
        self.s0.shape()
    }

    /// Save the statistics to a file
    /// # Arguments
    /// * `filename` - The name of the file to save the statistics to
    /// * `names` - The names of the statistics
    /// * `sep` - Separator for text formats
    /// * `row_column_name` - Name for the row column in parquet format (defaults to "stat")
    pub fn save(
        &self,
        filename: &str,
        names: &[Box<str>],
        sep: &str,
        row_column_name: Option<&str>,
    ) -> anyhow::Result<()> {
        match file_ext(filename).unwrap_or(Box::from("")).as_ref() {
            "parquet" => {
                let nnz = &self.count_positives();
                let tot = &self.sum();
                let mu = &self.mean();
                let sig = &self.std();

                let n = nnz.len();
                let nnz_col = nnz.clone().into_shape_with_order((n,)).unwrap();
                let tot_col = tot.clone().into_shape_with_order((n,)).unwrap();
                let mu_col = mu.clone().into_shape_with_order((n,)).unwrap();
                let sig_col = sig.clone().into_shape_with_order((n,)).unwrap();

                let stacked = stack(
                    Axis(1),
                    &[
                        nnz_col.view(),
                        tot_col.view(),
                        mu_col.view(),
                        sig_col.view(),
                    ],
                )
                .unwrap();

                let column_names: Vec<Box<str>> = vec!["nnz", "tot", "mu", "sig"]
                    .into_iter()
                    .map(|s| s.into())
                    .collect();

                let row_col = row_column_name.or(Some("stat"));
                stacked.to_parquet_with_names(
                    filename,
                    (Some(names), row_col),
                    Some(&column_names),
                )?;
            }
            _ => {
                let mut out = self.to_string_vec(names, sep)?;
                let header = format!("#name{}nnz{}tot{}mu{}sig", sep, sep, sep, sep);
                out.insert(0, header.into_boxed_str());
                write_lines(&out, filename)?;
            }
        };

        Ok(())
    }

    pub fn to_string_vec(&self, names: &[Box<str>], sep: &str) -> anyhow::Result<Vec<Box<str>>> {
        if names.len() != self.shape()[0] {
            anyhow::bail!(
                "The number of names does not match the number of the first dimension of the statistics"
            );
        }

        let nnz_: Vec<Box<str>> = to_string_vec(&self.count_positives(), sep);
        let tot_ = to_string_vec(&self.sum(), sep);
        let mu_: Vec<Box<str>> = to_string_vec(&self.mean(), sep);
        let sig_: Vec<Box<str>> = to_string_vec(&self.std(), sep);

        let out: Vec<Box<str>> = (0..self.shape()[0])
            .map(|i| {
                format!(
                    "{}{}{}{}{}{}{}{}{}",
                    names[i], sep, nnz_[i], sep, tot_[i], sep, mu_[i], sep, sig_[i]
                )
                .into_boxed_str()
            })
            .collect();
        Ok(out)
    }
}

impl<S> RunningStatOps<f32> for RunningStatistics<S>
where
    S: Dimension + RemoveAxis,
{
    type Output = ArrayBase<OwnedRepr<f32>, S>;

    fn clear(&mut self) {
        Self::clear(self)
    }

    fn count_positives(&self) -> Self::Output {
        Self::count_positives(self)
    }

    fn sum(&self) -> Self::Output {
        Self::sum(self)
    }

    fn mean(&self) -> Self::Output {
        Self::mean(self)
    }

    fn variance(&self) -> Self::Output {
        Self::variance(self)
    }

    fn std(&self) -> Self::Output {
        Self::std(self)
    }
}

fn to_string_vec<S>(xx: &ArrayBase<OwnedRepr<f32>, S>, sep: &str) -> Vec<Box<str>>
where
    S: Dimension + RemoveAxis,
{
    xx.axis_iter(Axis(0))
        .map(|m| {
            m.iter()
                .map(|v| {
                    if *v > 1e-4 {
                        format!("{:.4}", v)
                            .trim_end_matches('0')
                            .trim_end_matches('.')
                            .to_string()
                    } else if *v > 1e-20 {
                        format!("{:.4e}", v)
                    } else {
                        "0".to_string()
                    }
                })
                .collect::<Vec<String>>()
                .join(sep)
                .clone()
                .into_boxed_str()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, Ix1};

    /// An f32 running sum stops counting exactly at 2^24: 1.0 added to
    /// 16_777_216.0f32 is 16_777_216.0f32. The sums are f64, so they don't.
    #[test]
    fn f32_sums_stay_exact_past_two_to_the_24() {
        let n = (1usize << 24) + 10;
        let mut stat = RunningStatistics::new(Ix1(2));
        for _ in 0..n {
            stat.add_element(&0, 1.0);
        }
        // n is even, so it is representable in f32: the sum must hit it exactly.
        assert_eq!(stat.sum(), arr1(&[n as f32, 0.0]));
        assert_eq!(stat.count_positives(), arr1(&[n as f32, 0.0]));
        assert_eq!(stat.std(), arr1(&[0.0, 0.0]));
    }

    /// A large mean and a small spread: the one-pass variance in f32 cancels
    /// to noise (or below zero, giving a NaN sd); in f64 it holds.
    #[test]
    fn f32_variance_of_large_mean_small_spread() {
        let mut stat = RunningStatistics::new(Ix1(1));
        for j in 0..10_000 {
            let v = 1000.0f32 + if j % 2 == 0 { 0.5 } else { -0.5 };
            stat.add_element(&0, v);
        }
        let var = stat.variance()[0];
        assert!((var - 0.25).abs() < 1e-4, "variance {var}");
        assert!((stat.std()[0] - 0.5).abs() < 1e-4);
        let via_trait = <RunningStatistics<Ix1> as RunningStatOps<f32>>::variance(&stat);
        assert_eq!(via_trait, stat.variance());
    }

    /// A constant row has variance zero, never a negative rounding residue
    /// whose square root is NaN.
    #[test]
    fn constant_row_has_zero_sd() {
        let mut stat = RunningStatistics::new(Ix1(1));
        let x = arr1(&[0.1f32]);
        for _ in 0..1000 {
            stat.add(&x);
        }
        assert!(stat.variance()[0] >= 0.0);
        assert!(!stat.std()[0].is_nan());
        let sd = <RunningStatistics<Ix1> as RunningStatOps<f32>>::std(&stat);
        assert!(!sd[0].is_nan());
    }

    /// Non-finite values are skipped: they add to no count and no sum.
    #[test]
    fn non_finite_values_are_skipped() {
        let mut stat = RunningStatistics::new(Ix1(2));
        stat.add(&arr1(&[2.0f32, f32::NAN]));
        stat.add(&arr1(&[4.0f32, f32::INFINITY]));
        assert_eq!(stat.count_positives(), arr1(&[2.0, 0.0]));
        assert_eq!(stat.sum(), arr1(&[6.0, 0.0]));
        assert!((stat.mean()[0] - 3.0).abs() < 1e-6);
        assert_eq!(stat.mean()[1], 0.0);
        assert!((stat.variance()[0] - 1.0).abs() < 1e-6);
    }

    /// The `RunningStatOps` impl forwards to the inherent methods.
    #[test]
    fn trait_matches_inherent() {
        type Stat = RunningStatistics<Ix1>;
        let mut stat = Stat::new(Ix1(2));
        stat.add(&arr1(&[1.0f32, -2.0]));
        stat.add(&arr1(&[3.0f32, 0.0]));
        assert_eq!(
            RunningStatOps::count_positives(&stat),
            Stat::count_positives(&stat)
        );
        assert_eq!(RunningStatOps::sum(&stat), Stat::sum(&stat));
        assert_eq!(RunningStatOps::mean(&stat), Stat::mean(&stat));
        assert_eq!(RunningStatOps::variance(&stat), Stat::variance(&stat));
        assert_eq!(RunningStatOps::std(&stat), Stat::std(&stat));
        RunningStatOps::clear(&mut stat);
        assert_eq!(stat.sum(), arr1(&[0.0, 0.0]));
    }
}
