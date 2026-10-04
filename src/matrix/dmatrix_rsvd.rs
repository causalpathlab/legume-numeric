use crate::matrix::traits::*;
use nalgebra::{DMatrix, DVector};
use nalgebra_sparse::{csc::CscMatrix, csr::CsrMatrix};
use rayon::prelude::*;
use std::borrow::Cow;

/// Fixed start-vector seed for the randomized-SVD subspace iteration. The
/// iteration converges onto the dominant subspace, so pinning the start makes
/// `rsvd` reproducible without altering the subspace it recovers.
const RSVD_SUBSPACE_SEED: u64 = 0x5253_5644_5342_5350; // "RSVDSBSP"

/// Compute the Nystrom basis: `U * diag(1 / (s + eps))`.
///
/// Given the left singular vectors `u` and singular values `s` from an SVD,
/// returns the pseudo-inverted projection matrix used for out-of-sample
/// Nystrom extension.
pub fn nystrom_basis(u: &DMatrix<f32>, s: &DVector<f32>) -> DMatrix<f32> {
    let eps = 1e-8;
    let sinv = DVector::from_iterator(s.len(), s.iter().map(|&si| 1.0 / (si + eps)));
    u * DMatrix::from_diagonal(&sinv)
}

/// What the subspace iteration needs of `X`: products with `X` and `Xᵀ`.
trait LinOp<T: nalgebra::Scalar> {
    fn matmul(&self, other: &DMatrix<T>) -> DMatrix<T>;
    fn transpose_matmul(&self, other: &DMatrix<T>) -> DMatrix<T>;
    fn num_rows(&self) -> usize;
    fn num_columns(&self) -> usize;
}

impl<T> LinOp<T> for DMatrix<T>
where
    T: nalgebra::RealField + num_traits::Float + Copy,
{
    fn matmul(&self, other: &DMatrix<T>) -> DMatrix<T> {
        self * other
    }

    fn transpose_matmul(&self, other: &DMatrix<T>) -> DMatrix<T> {
        self.transpose() * other
    }

    fn num_rows(&self) -> usize {
        self.nrows()
    }
    fn num_columns(&self) -> usize {
        self.ncols()
    }
}

/// A sparse matrix held in both orientations for the subspace iteration,
/// which multiplies by `X` and by `Xᵀ` in turn. Compressed rows of `X` give
/// `X · B` and compressed columns, the rows of `Xᵀ`, give `Xᵀ · B`, each
/// row-parallel with no transpose per product. The orientation the caller
/// does not hold is built once.
struct SparseOp<'a, T: nalgebra::Scalar> {
    csr: Cow<'a, CsrMatrix<T>>,
    csc: Cow<'a, CscMatrix<T>>,
}

impl<'a, T> SparseOp<'a, T>
where
    T: nalgebra::RealField + Copy,
{
    fn from_csr(x: &'a CsrMatrix<T>) -> Self {
        Self {
            csr: Cow::Borrowed(x),
            csc: Cow::Owned(CscMatrix::from(x)),
        }
    }

    fn from_csc(x: &'a CscMatrix<T>) -> Self {
        Self {
            csr: Cow::Owned(CsrMatrix::from(x)),
            csc: Cow::Borrowed(x),
        }
    }
}

impl<T> LinOp<T> for SparseOp<'_, T>
where
    T: nalgebra::RealField + Copy,
{
    fn matmul(&self, other: &DMatrix<T>) -> DMatrix<T> {
        let x = &*self.csr;
        rows_times_dense(x.row_offsets(), x.col_indices(), x.values(), other)
    }
    fn transpose_matmul(&self, other: &DMatrix<T>) -> DMatrix<T> {
        let x = &*self.csc;
        rows_times_dense(x.col_offsets(), x.row_indices(), x.values(), other)
    }
    fn num_rows(&self) -> usize {
        self.csr.nrows()
    }
    fn num_columns(&self) -> usize {
        self.csr.ncols()
    }
}

/// Rows below which a thread is not worth splitting off.
const MIN_ROWS_PER_TASK: usize = 256;

/// Output columns a row accumulates at once, in registers rather than in the
/// output row, which would be reloaded and stored once per entry.
const COLUMN_BLOCK: usize = 8;

/// `S · B` for `S` given by compressed rows (`offsets`, `indices`, `values`),
/// one row of the product per task. `B` is read through its transpose, so a
/// row of `B` is contiguous; each output entry sums in stored order, so the
/// result does not depend on the thread count. The product is bound by
/// gathering rows of `B`, not by arithmetic, so wider SIMD buys nothing.
fn rows_times_dense<T>(
    offsets: &[usize],
    indices: &[usize],
    values: &[T],
    b: &DMatrix<T>,
) -> DMatrix<T>
where
    T: nalgebra::RealField + Copy,
{
    let m = offsets.len() - 1;
    let c = b.ncols();
    if c == 0 {
        return DMatrix::zeros(m, 0);
    }
    let bt = b.transpose();
    let bt = bt.as_slice();
    let mut out = vec![T::zero(); m * c];
    out.par_chunks_mut(c)
        .enumerate()
        .with_min_len(MIN_ROWS_PER_TASK)
        .for_each(|(i, row)| {
            let entries = offsets[i]..offsets[i + 1];
            let (idx, val) = (&indices[entries.clone()], &values[entries]);
            let mut k0 = 0;
            while k0 + COLUMN_BLOCK <= c {
                let mut acc = [T::zero(); COLUMN_BLOCK];
                for (&j, &v) in idx.iter().zip(val) {
                    let src = &bt[j * c + k0..j * c + k0 + COLUMN_BLOCK];
                    for (a, &s) in acc.iter_mut().zip(src) {
                        *a += v * s;
                    }
                }
                row[k0..k0 + COLUMN_BLOCK].copy_from_slice(&acc);
                k0 += COLUMN_BLOCK;
            }
            for (&j, &v) in idx.iter().zip(val) {
                for (o, &s) in row[k0..].iter_mut().zip(&bt[j * c + k0..(j + 1) * c]) {
                    *o += v * s;
                }
            }
        });
    DMatrix::from_row_slice(m, c, &out)
}

fn _subspace_iteration<T, D>(
    xx: &D,
    rank_and_oversample: usize,
    power_iters: usize,
) -> anyhow::Result<DMatrix<T>>
where
    T: nalgebra::RealField + num_traits::Float + Copy,
    D: LinOp<T>,
{
    let nc = xx.num_columns();
    // Fixed seed: with enough power iterations the subspace converges onto
    // the dominant one regardless of the start, so a pinned (rather than
    // entropy) draw makes the whole randomized SVD reproducible run-to-run —
    // which in turn pins every downstream consumer (binary-sketch collapse,
    // layout, SVD fits) — without changing what subspace it recovers. With
    // very few iterations the result does depend on this start.
    let mut qq = DMatrix::<T>::runif_seeded(nc, rank_and_oversample, RSVD_SUBSPACE_SEED);
    let half = T::from(0.5).expect("no half found");
    qq.iter_mut().for_each(|x| *x -= half);

    // Each half-step re-orthonormalises the iterate with a thin QR. The
    // basis must span exactly the range of the product it came from: a
    // pivoted LU factor does not (its permutation is lost), and iterating on
    // a row-permuted range does not converge onto the dominant subspace.
    for _i in 0..power_iters {
        let ll = xx.matmul(&qq).qr().q();
        qq = xx.transpose_matmul(&ll).qr().q();
    }

    let qr_q = xx.matmul(&qq).qr().q();
    let kk = rank_and_oversample.min(qr_q.ncols());
    let ret = qr_q.columns(0, kk).into_owned();

    Ok(ret)
}

fn _randomized_svd<T, D>(
    xx: &D,
    max_rank: usize,
    args: &RsvdArgs,
) -> anyhow::Result<(DMatrix<T>, DVector<T>, DMatrix<T>)>
where
    T: nalgebra::RealField + num_traits::Float + Copy,
    D: LinOp<T>,
{
    let nr = xx.num_rows();
    let nc = xx.num_columns();

    let mut rank = nr.min(nc);
    let mut oversample = 0;

    if max_rank > 0 && rank > max_rank {
        rank = max_rank;
        oversample = args.oversample;
    }

    anyhow::ensure!(rank > 0, "randomized SVD of an empty {nr} x {nc} matrix");

    // Keep the oversampled basis through the projection: its columns are
    // not ordered by singular value, so truncating here would discard part
    // of the dominant subspace. The rank is applied to the small SVD below.
    // Columns beyond the matrix's own rank add nothing; the cap sits above
    // anything the default oversampling reaches, so it changes no default
    // result and only stops an oversized `oversample` from overflowing or
    // allocating for nothing.
    let width = rank
        .saturating_add(oversample)
        .min(nr.min(nc) + RsvdArgs::default().oversample);
    let qq = _subspace_iteration(xx, width, args.power_iters)?;
    let rank = rank.min(qq.ncols());

    // bb = qqᵀ · xx
    let bb = xx.transpose_matmul(&qq).transpose();

    let svd = bb.svd(true, true);

    if let (Some(svd_u), Some(svd_vt)) = (svd.u, svd.v_t) {
        return Ok((
            &qq * svd_u.columns(0, rank),
            svd.singular_values.rows(0, rank).into_owned(),
            svd_vt.transpose().columns(0, rank).into_owned(),
        ));
    }
    Err(anyhow::anyhow!("randomized SVD failed"))
}

impl<T> RandomizedAlgs for DMatrix<T>
where
    T: nalgebra::RealField + num_traits::Float + Copy,
{
    type InMat = DMatrix<T>;
    type OutMat = DMatrix<T>;
    type DVec = DVector<T>;
    type Scalar = T;

    fn rsvd_with(
        &self,
        max_rank: usize,
        args: &RsvdArgs,
    ) -> anyhow::Result<(Self::OutMat, Self::DVec, Self::OutMat)> {
        _randomized_svd(self, max_rank, args)
    }
}

impl<T> RandomizedAlgs for CscMatrix<T>
where
    T: nalgebra::RealField + num_traits::Float + Copy,
{
    type InMat = CscMatrix<T>;
    type OutMat = DMatrix<T>;
    type DVec = DVector<T>;
    type Scalar = T;

    fn rsvd_with(
        &self,
        max_rank: usize,
        args: &RsvdArgs,
    ) -> anyhow::Result<(Self::OutMat, Self::DVec, Self::OutMat)> {
        _randomized_svd(&SparseOp::from_csc(self), max_rank, args)
    }
}

impl<T> RandomizedAlgs for CsrMatrix<T>
where
    T: nalgebra::RealField + num_traits::Float + Copy,
{
    type InMat = CsrMatrix<T>;
    type OutMat = DMatrix<T>;
    type DVec = DVector<T>;
    type Scalar = T;

    fn rsvd_with(
        &self,
        max_rank: usize,
        args: &RsvdArgs,
    ) -> anyhow::Result<(Self::OutMat, Self::DVec, Self::OutMat)> {
        _randomized_svd(&SparseOp::from_csr(self), max_rank, args)
    }
}

#[cfg(test)]
#[path = "dmatrix_rsvd_tests.rs"]
mod tests;
