//! A cells-by-features count matrix read column by column, for code that
//! streams counts once (samplers, graph builders) without depending on any
//! storage crate. data-beans implements it for its backends; a
//! [`CscMatrix`] implements it here, for tests and small in-memory data.

use nalgebra_sparse::CscMatrix;
use std::ops::Range;

/// Counts of `n_cells()` cells over `n_features()` features, streamed in
/// slabs of consecutive cells.
pub trait CellCountStream: Sync {
    fn n_cells(&self) -> usize;
    fn n_features(&self) -> usize;
    /// Every nonzero of the whole matrix, when known: the caller sizes its
    /// slabs from it.
    fn nnz_hint(&self) -> Option<usize>;
    /// Call `f(feature, cell - cells.start, count)` for each stored count of
    /// the cells `cells`, feature on this stream's axis (`< n_features()`).
    /// A feature may repeat within a cell (each call adds to its count); the
    /// order within a cell is the storage's own.
    fn for_each_count(
        &self,
        cells: Range<usize>,
        f: &mut dyn FnMut(u32, u32, f32),
    ) -> anyhow::Result<()>;
}

impl CellCountStream for CscMatrix<f32> {
    fn n_cells(&self) -> usize {
        self.ncols()
    }

    fn n_features(&self) -> usize {
        self.nrows()
    }

    fn nnz_hint(&self) -> Option<usize> {
        Some(self.nnz())
    }

    fn for_each_count(
        &self,
        cells: Range<usize>,
        f: &mut dyn FnMut(u32, u32, f32),
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            cells.end <= self.ncols(),
            "cells {cells:?} past the matrix's {} columns",
            self.ncols()
        );
        for (local, c) in cells.enumerate() {
            let col = self.col(c);
            for (&r, &v) in col.row_indices().iter().zip(col.values()) {
                f(r as u32, local as u32, v);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra_sparse::CooMatrix;

    fn three_by_four() -> CscMatrix<f32> {
        let mut coo = CooMatrix::new(3, 4);
        coo.push(0, 0, 1.0);
        coo.push(2, 0, 2.0);
        coo.push(1, 2, 3.0);
        coo.push(2, 3, 4.0);
        CscMatrix::from(&coo)
    }

    /// Cells are reported relative to the slab's first cell, with every
    /// stored count, and nothing outside the slab.
    #[test]
    fn a_slab_reports_its_own_counts_with_local_cells() {
        let m = three_by_four();
        let mut seen = Vec::new();
        m.for_each_count(1..4, &mut |f, c, v| seen.push((f, c, v)))
            .expect("in range");
        assert_eq!(seen, vec![(1, 1, 3.0), (2, 2, 4.0)]);
        assert_eq!((m.n_cells(), m.n_features(), m.nnz_hint()), (4, 3, Some(4)));
    }

    #[test]
    fn a_slab_past_the_last_cell_is_refused() {
        let m = three_by_four();
        assert!(m.for_each_count(2..5, &mut |_, _, _| {}).is_err());
    }
}
