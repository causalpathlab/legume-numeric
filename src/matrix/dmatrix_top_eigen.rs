//! Leading algebraic eigenpairs of a symmetric matrix by Chebyshev-filtered
//! block subspace iteration; see [`SymmetricTopEigen`].

use crate::matrix::rand_util::{mix_seed, name_seed};
use crate::matrix::traits::*;
use nalgebra::{DMatrix, DVector};
use num_traits::Float;
use rayon::prelude::*;

/// Fixed seed for the random start columns and the Lanczos start vector, so
/// a call is reproducible run to run.
const TOP_EIGEN_SEED: u64 = 0x544F_5045_4947_454E; // "TOPEIGEN"

/// Lanczos steps used to bound the spectrum before iterating.
const LANCZOS_STEPS: usize = 40;

/// Fraction of the spectrum's width the damped interval reaches below the
/// Lanczos estimate of `λ_min`.
const LOWER_MARGIN: f64 = 0.05;

/// Rows of the product one thread computes, at least.
const MIN_ROWS_PER_TASK: usize = 64;

/// The scalar the solver works in.
trait EigScalar: nalgebra::RealField + Float + Copy + Send + Sync {}
impl<T: nalgebra::RealField + Float + Copy + Send + Sync> EigScalar for T {}

fn to_f64<T: EigScalar>(x: T) -> f64 {
    num_traits::ToPrimitive::to_f64(&x).unwrap_or(f64::NAN)
}

fn from_f64<T: EigScalar>(x: f64) -> T {
    <T as num_traits::NumCast>::from(x).expect("f64 fits the scalar")
}

/// What the solver needs of the symmetric matrix `A`.
trait SymOp<T: EigScalar> {
    fn dim(&self) -> usize;
    /// `A · X`.
    fn apply(&self, x: &DMatrix<T>) -> anyhow::Result<DMatrix<T>>;
    /// `A` itself, for the dense fallback.
    fn dense(&self) -> anyhow::Result<DMatrix<T>>;
}

/// `A · X`, one block of output rows per task, each a blocked GEMM of
/// `A[j..j+w, :]` with `X` (nalgebra's own product is single-threaded).
fn sym_times<T: EigScalar>(a: &DMatrix<T>, x: &DMatrix<T>) -> DMatrix<T> {
    let n = a.ncols();
    let block = n
        .div_ceil(rayon::current_num_threads())
        .max(MIN_ROWS_PER_TASK);
    let starts: Vec<usize> = (0..n).step_by(block).collect();
    let parts: Vec<DMatrix<T>> = starts
        .par_iter()
        .map(|&j| a.rows(j, block.min(n - j)) * x)
        .collect();
    let mut out = DMatrix::zeros(n, x.ncols());
    for (&j, part) in starts.iter().zip(&parts) {
        out.rows_mut(j, part.nrows()).copy_from(part);
    }
    out
}

impl<T: EigScalar> SymOp<T> for DMatrix<T> {
    fn dim(&self) -> usize {
        self.nrows()
    }
    fn apply(&self, x: &DMatrix<T>) -> anyhow::Result<DMatrix<T>> {
        Ok(sym_times(self, x))
    }
    fn dense(&self) -> anyhow::Result<DMatrix<T>> {
        Ok(self.clone())
    }
}

/// Eigenpairs of a small symmetric matrix, sorted by descending value.
fn sorted_eigen<T: EigScalar>(h: DMatrix<T>) -> (DVector<T>, DMatrix<T>) {
    let eig = h.symmetric_eigen();
    let mut order: Vec<usize> = (0..eig.eigenvalues.len()).collect();
    order.sort_by(|&i, &j| {
        eig.eigenvalues[j]
            .partial_cmp(&eig.eigenvalues[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let values = DVector::from_iterator(order.len(), order.iter().map(|&i| eig.eigenvalues[i]));
    let vectors = DMatrix::from_columns(
        &order
            .iter()
            .map(|&i| eig.eigenvectors.column(i))
            .collect::<Vec<_>>(),
    );
    (values, vectors)
}

/// Remove from `w` its components along the orthonormal columns of `basis`,
/// twice, as one pass of classical Gram–Schmidt loses orthogonality.
fn project_out<T: EigScalar>(w: &mut DMatrix<T>, basis: nalgebra::DMatrixView<T>) {
    for _ in 0..2 {
        let coef = basis.tr_mul(w);
        *w -= basis * coef;
    }
}

/// Lower and upper bounds on the spectrum of `A` from a Lanczos run with full
/// reorthogonalisation. Each extreme Ritz value `θ` is widened by its
/// residual `|β s|`, which an eigenvalue lies within; Lanczos finds the
/// extremes first, so these bracket the spectrum. On breakdown (an invariant
/// Krylov subspace) the run continues from a fresh orthogonal vector, which
/// keeps the tridiagonal's eigenvalues eigenvalues of `A`.
fn lanczos_bounds<T: EigScalar, D: SymOp<T>>(op: &D) -> anyhow::Result<(f64, f64)> {
    let n = op.dim();
    let steps = LANCZOS_STEPS.min(n);
    let mut basis = DMatrix::<T>::zeros(n, steps);
    let mut alpha = vec![0.0; steps];
    let mut beta = vec![0.0; steps];
    let mut v = DMatrix::<T>::rnorm_seeded(n, 1, name_seed(TOP_EIGEN_SEED, "lanczos"));
    let mut len = 0;
    // Largest `‖A v‖` seen, a lower estimate of `‖A‖`, to judge breakdown by.
    let mut scale = 0.0f64;
    while len < steps {
        let nrm = v.norm();
        if to_f64(nrm) <= f64::MIN_POSITIVE {
            break;
        }
        v /= nrm;
        basis.set_column(len, &v.column(0));
        let mut w = op.apply(&v)?;
        alpha[len] = to_f64(v.column(0).dot(&w.column(0)));
        scale = scale.max(to_f64(w.norm()));
        anyhow::ensure!(
            alpha[len].is_finite() && scale.is_finite(),
            "top_eigen: the matrix has non-finite entries"
        );
        // Full reorthogonalisation against every basis vector.
        project_out(&mut w, basis.columns(0, len + 1));
        len += 1;
        let b = to_f64(w.norm());
        let breakdown = b <= 1e3 * to_f64(T::epsilon()) * scale.max(f64::MIN_POSITIVE);
        beta[len - 1] = if breakdown { 0.0 } else { b };
        if breakdown && len < steps {
            // Restart orthogonally to the basis, with β = 0.
            let seed = mix_seed(name_seed(TOP_EIGEN_SEED, "lanczos"), len as u64);
            w = DMatrix::<T>::rnorm_seeded(n, 1, seed);
            project_out(&mut w, basis.columns(0, len));
        }
        v = w;
    }
    anyhow::ensure!(len > 0, "Lanczos found no direction");
    let mut t = DMatrix::<f64>::zeros(len, len);
    for i in 0..len {
        t[(i, i)] = alpha[i];
        if i + 1 < len {
            t[(i, i + 1)] = beta[i];
            t[(i + 1, i)] = beta[i];
        }
    }
    let (theta, s) = sorted_eigen(t);
    let resid = |i: usize| (beta[len - 1] * s[(len - 1, i)]).abs();
    let hi = theta[0] + resid(0);
    let lo = theta[len - 1] - resid(len - 1);
    Ok((lo, hi))
}

/// The block after Rayleigh–Ritz: orthonormal Ritz vectors `v`, `A v`, and
/// the Ritz values in descending order.
struct Ritz<T: nalgebra::Scalar> {
    v: DMatrix<T>,
    av: DMatrix<T>,
    theta: DVector<T>,
}

/// Orthonormalise `x` and project `A` onto its span.
fn rayleigh_ritz<T: EigScalar, D: SymOp<T>>(op: &D, x: DMatrix<T>) -> anyhow::Result<Ritz<T>> {
    let q = x.qr().q();
    let aq = op.apply(&q)?;
    let h = q.transpose() * &aq;
    let h = (&h + h.transpose()) * from_f64::<T>(0.5);
    let (theta, s) = sorted_eigen(h);
    Ok(Ritz {
        v: &q * &s,
        av: &aq * &s,
        theta,
    })
}

/// `p(A) · V` for the degree-`m` Chebyshev polynomial that is bounded by 1
/// on `[c − e, c + e]` and is 1 at `c + e τ`, `τ > 1`, by the scaled
/// three-term recurrence (Zhou and Saad 2007), so nothing overflows. `av`
/// is `A V`, which saves the first product.
fn chebyshev_filter<T: EigScalar, D: SymOp<T>>(
    op: &D,
    ritz: &Ritz<T>,
    m: usize,
    c: f64,
    e: f64,
    tau: f64,
) -> anyhow::Result<DMatrix<T>> {
    let mut sigma = 1.0 / tau;
    let mut prev = ritz.v.clone();
    let mut cur = (&ritz.av - &ritz.v * from_f64::<T>(c)) * from_f64::<T>(sigma / e);
    for _ in 1..m {
        let next_sigma = 1.0 / (2.0 * tau - sigma);
        let mut next = op.apply(&cur)?;
        next -= &cur * from_f64::<T>(c);
        next *= from_f64::<T>(2.0 * next_sigma / e);
        next -= prev * from_f64::<T>(sigma * next_sigma);
        prev = cur;
        cur = next;
        sigma = next_sigma;
    }
    Ok(cur)
}

/// Flip each column so its largest-magnitude entry is positive, which pins
/// the sign an eigenvector is otherwise free to take.
fn fix_signs<T: EigScalar>(v: &mut DMatrix<T>) {
    for mut col in v.column_iter_mut() {
        if col.nrows() > 0 && col[col.iamax()] < T::zero() {
            col.neg_mut();
        }
    }
}

/// The first `rank` pairs, signs fixed.
fn leading<T: EigScalar>(
    values: &DVector<T>,
    vectors: &DMatrix<T>,
    rank: usize,
    iters: usize,
    converged: bool,
) -> TopEigen<DMatrix<T>, DVector<T>> {
    let mut vectors = vectors.columns(0, rank).into_owned();
    fix_signs(&mut vectors);
    TopEigen {
        values: values.rows(0, rank).into_owned(),
        vectors,
        iters,
        converged,
    }
}

/// The top `rank` pairs of `op`.
fn top_eigen_op<T: EigScalar, D: SymOp<T>>(
    op: &D,
    rank: usize,
    args: &TopEigenArgs,
    warm_start: Option<&DMatrix<T>>,
) -> anyhow::Result<TopEigen<DMatrix<T>, DVector<T>>> {
    let n = op.dim();
    let rank = rank.min(n);
    if rank == 0 {
        return Ok(leading(
            &DVector::zeros(0),
            &DMatrix::zeros(n, 0),
            0,
            0,
            true,
        ));
    }
    let k = rank.saturating_add(args.oversample).min(n);

    // A block over half the dimension costs more per cycle than a dense
    // solve does in all.
    if 2 * k >= n {
        let (values, vectors) = sorted_eigen(op.dense()?);
        return Ok(leading(&values, &vectors, rank, 0, true));
    }

    let (mut lo, hi) = lanczos_bounds(op)?;
    let norm = lo.abs().max(hi.abs()).max(f64::MIN_POSITIVE);
    // Keep the block's conditioning within √ε: a column amplified past that
    // relative to another loses the other to rounding in the QR.
    let growth_cap = (2.0 / to_f64(T::epsilon()).sqrt()).ln();

    let mut x = DMatrix::<T>::rnorm_seeded(n, k, TOP_EIGEN_SEED);
    if let Some(w) = warm_start {
        anyhow::ensure!(
            w.nrows() == n,
            "warm start has {} rows, the matrix {n}",
            w.nrows()
        );
        let wc = w.ncols().min(k);
        x.columns_mut(0, wc).copy_from(&w.columns(0, wc));
    }
    let mut ritz = rayleigh_ritz(op, x)?;
    let mut iters = 0;
    let converged = loop {
        let worst = (0..rank)
            .map(|i| to_f64((ritz.av.column(i) - ritz.v.column(i) * ritz.theta[i]).norm()))
            .fold(0.0f64, f64::max);
        if worst <= args.tol * norm {
            break true;
        }
        if iters >= args.max_iters {
            break false;
        }

        // Damp [a, b], everything up to the block's lowest Ritz value, and
        // amplify above it, scaled to 1 at the top of the spectrum.
        // The Lanczos bound on λ_min is an estimate, and a component below
        // `a` would be amplified, not damped, so leave a margin: widening
        // the interval by 5% costs little convergence.
        let (t_hi, t_lo) = (to_f64(ritz.theta[0]), to_f64(ritz.theta[k - 1]));
        lo = lo.min(t_lo);
        let a = lo - LOWER_MARGIN * (hi - lo).max(norm * f64::EPSILON);
        let b = t_lo.max(a + 1e-3 * (hi.max(t_hi) - a));
        let top = hi.max(t_hi).max(b + (b - a));
        let (c, e) = ((b + a) / 2.0, (b - a) / 2.0);
        let tau = (top - c) / e;
        let lead = ((t_hi - c) / e).max(1.0 + 1e-12);
        let degree =
            ((growth_cap / lead.acosh()).floor() as usize).clamp(1, args.max_degree.max(1));

        let x = chebyshev_filter(op, &ritz, degree, c, e, tau)?;
        ritz = rayleigh_ritz(op, x)?;
        iters += 1;
    };

    if !converged {
        log::warn!(
            "top_eigen: {rank} pairs not converged to tol {} in {iters} iterations",
            args.tol
        );
    }
    Ok(leading(&ritz.theta, &ritz.v, rank, iters, converged))
}

// The bound is `EigScalar` spelled out: a private trait cannot bound a
// public impl.
impl<T> SymmetricTopEigen for DMatrix<T>
where
    T: nalgebra::RealField + Float + Copy + Send + Sync,
{
    type Mat = DMatrix<T>;
    type DVec = DVector<T>;

    fn top_eigen_with(
        &self,
        rank: usize,
        args: &TopEigenArgs,
        warm_start: Option<&Self::Mat>,
    ) -> anyhow::Result<TopEigen<Self::Mat, Self::DVec>> {
        anyhow::ensure!(
            self.is_square(),
            "top_eigen needs a square matrix, got {} x {}",
            self.nrows(),
            self.ncols()
        );
        top_eigen_op(self, rank, args, warm_start)
    }
}

#[cfg(feature = "tensor")]
mod tensor {
    use super::*;
    use candle_core::{DType, Tensor};

    /// A 2-D tensor as the operator: products run in candle, on the tensor's
    /// own device and threads, and only the `n × k` block crosses over.
    struct TensorOp<'a>(&'a Tensor);

    impl<T> SymOp<T> for TensorOp<'_>
    where
        T: EigScalar + candle_core::WithDType,
    {
        fn dim(&self) -> usize {
            self.0.dims()[0]
        }
        /// The column-major `X` is, read row-major, `Xᵀ`; `Xᵀ A = (A X)ᵀ`
        /// by symmetry, and its row-major buffer is `A X` column-major, so
        /// no transposes are needed either way.
        fn apply(&self, x: &DMatrix<T>) -> anyhow::Result<DMatrix<T>> {
            let (n, k) = x.shape();
            let xt = Tensor::from_slice(x.as_slice(), (k, n), self.0.device())?;
            let axt = xt.matmul(self.0)?;
            Ok(DMatrix::from_vec(n, k, axt.flatten_all()?.to_vec1()?))
        }
        fn dense(&self) -> anyhow::Result<DMatrix<T>> {
            DMatrix::<T>::from_tensor(self.0)
        }
    }

    fn run<T>(
        a: &Tensor,
        rank: usize,
        args: &TopEigenArgs,
        warm_start: Option<&Tensor>,
    ) -> anyhow::Result<TopEigen<Tensor, Tensor>>
    where
        T: EigScalar + candle_core::WithDType,
    {
        let warm = warm_start
            .map(|w| DMatrix::<T>::from_tensor(&w.to_dtype(a.dtype())?))
            .transpose()?;
        let out = top_eigen_op::<T, _>(&TensorOp(a), rank, args, warm.as_ref())?;
        Ok(TopEigen {
            values: Tensor::from_slice(out.values.as_slice(), out.values.len(), a.device())?,
            vectors: out.vectors.to_tensor(a.device())?.contiguous()?,
            iters: out.iters,
            converged: out.converged,
        })
    }

    impl SymmetricTopEigen for Tensor {
        type Mat = Tensor;
        type DVec = Tensor;

        /// Runs in the tensor's dtype (`f32` or `f64`); the results are
        /// tensors of that dtype on its device.
        fn top_eigen_with(
            &self,
            rank: usize,
            args: &TopEigenArgs,
            warm_start: Option<&Self::Mat>,
        ) -> anyhow::Result<TopEigen<Self::Mat, Self::DVec>> {
            let dims = self.dims();
            anyhow::ensure!(
                dims.len() == 2 && dims[0] == dims[1],
                "top_eigen needs a square 2-D tensor, got {dims:?}"
            );
            match self.dtype() {
                DType::F32 => run::<f32>(self, rank, args, warm_start),
                DType::F64 => run::<f64>(self, rank, args, warm_start),
                dt => anyhow::bail!("top_eigen supports f32 and f64 tensors, got {dt:?}"),
            }
        }
    }
}

#[cfg(feature = "ndarray")]
mod ndarray_impl {
    use super::*;
    use ndarray::{Array1, Array2, ShapeBuilder};

    fn to_dmatrix<T: EigScalar>(a: &Array2<T>) -> DMatrix<T> {
        let (n, m) = a.dim();
        DMatrix::from_fn(n, m, |i, j| a[(i, j)])
    }

    fn to_array<T: EigScalar>(m: &DMatrix<T>) -> Array2<T> {
        // Column-major storage is Fortran order.
        Array2::from_shape_vec(m.shape().f(), m.as_slice().to_vec())
            .expect("shape matches the buffer")
    }

    /// Copies the matrix into a `DMatrix` once (`n²`, small next to the
    /// iteration) and runs the dense solver: its products are parallel,
    /// and ndarray's `blas` feature links no BLAS of its own.
    // The bound is `EigScalar` spelled out: a private trait cannot bound a
    // public impl.
    impl<T> SymmetricTopEigen for Array2<T>
    where
        T: nalgebra::RealField + Float + Copy + Send + Sync,
    {
        type Mat = Array2<T>;
        type DVec = Array1<T>;

        fn top_eigen_with(
            &self,
            rank: usize,
            args: &TopEigenArgs,
            warm_start: Option<&Self::Mat>,
        ) -> anyhow::Result<TopEigen<Self::Mat, Self::DVec>> {
            let warm = warm_start.map(to_dmatrix);
            let out = to_dmatrix(self).top_eigen_with(rank, args, warm.as_ref())?;
            Ok(TopEigen {
                values: Array1::from_vec(out.values.as_slice().to_vec()),
                vectors: to_array(&out.vectors),
                iters: out.iters,
                converged: out.converged,
            })
        }
    }
}

#[cfg(test)]
#[path = "dmatrix_top_eigen_tests.rs"]
mod tests;
