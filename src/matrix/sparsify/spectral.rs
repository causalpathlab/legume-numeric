//! The Laplacian spectra DSE compares, the Diffusion Spectral Error Φ
//! between them, and Φ's gradient in every edge weight of the sparse graph.
//!
//! With `(λᵢ, vᵢ)` the eigenpairs of the full graph's Laplacian `L` and
//! `(λ̃ⱼ, ṽⱼ)` those of the sparse graph's `L̃`, zero modes left out,
//!
//! ```text
//! Φ = Σᵢⱼ ⟨vᵢ, ṽⱼ⟩² (λᵢ − λ̃ⱼ)² / (2 λᵢ λ̃ⱼ (λᵢ + λ̃ⱼ))
//!   = Σⱼ 1/(2λ̃ⱼ) + Σᵢ 1/(2λᵢ) − 2 Σᵢⱼ ⟨vᵢ, ṽⱼ⟩² / (λᵢ + λ̃ⱼ),
//! ```
//!
//! the expected squared distance between diffusion trajectories
//! `e^{−Lt} x₀` and `e^{−L̃t} x₀`, integrated over time, from `x₀ ~ N(0, I)`
//! (Fitz-Gibbon et al. 2026, Eq. 6). Each mode is weighed by 1/λ: the slow
//! modes, which the long-run diffusion is made of, count most.
//!
//! The gradient is exact, not a first-order guess per edge: the first sum
//! is a trace function of `L̃`, and the cross term's derivative, through the
//! Daleckii–Krein formula, collapses to a square:
//!
//! ```text
//! ∂Φ/∂w̃_ab = 2 Σᵢ (U_ai − U_bi)² − ½ ‖X_a − X_b‖²,
//! U = Ṽ K,  K_ji = ⟨vᵢ, ṽⱼ⟩ / (λ̃ⱼ + λᵢ),   X = Ṽ diag(1/λ̃),
//! ```
//!
//! valid for an edge in the sparse graph and for one not in it (its weight
//! 0), so adding, removing and reweighting edges are all ranked from one
//! eigendecomposition of each graph.

use nalgebra::{DMatrix, DVector, SymmetricEigen};

/// An edge `(a, b, w)` on a piece's local node ids.
pub(super) type Edge = (usize, usize, f64);

/// The Laplacian `D − A` of the `n`-node graph `edges`, dense.
pub(super) fn laplacian(n: usize, edges: &[Edge]) -> DMatrix<f64> {
    let mut l = DMatrix::<f64>::zeros(n, n);
    for &(a, b, w) in edges {
        l[(a, a)] += w;
        l[(b, b)] += w;
        l[(a, b)] -= w;
        l[(b, a)] -= w;
    }
    l
}

/// A connected graph's Laplacian eigenpairs without its zero mode.
#[derive(Clone)]
pub(super) struct Spectrum {
    /// The nonzero eigenvalues, ascending.
    pub values: DVector<f64>,
    /// Their eigenvectors, as columns.
    pub vectors: DMatrix<f64>,
}

impl Spectrum {
    /// The spectrum of the `n`-node graph `edges`, which must be connected
    /// (its smallest eigenvalue is the one zero mode dropped).
    pub fn of(n: usize, edges: &[Edge]) -> Self {
        let eig = SymmetricEigen::new(laplacian(n, edges));
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&i, &j| eig.eigenvalues[i].total_cmp(&eig.eigenvalues[j]));
        let kept = &order[1..];
        Self {
            // A connected graph's other eigenvalues are positive; the floor
            // only guards round-off.
            values: DVector::from_iterator(
                kept.len(),
                kept.iter().map(|&i| eig.eigenvalues[i].max(f64::EPSILON)),
            ),
            vectors: DMatrix::from_fn(n, kept.len(), |r, c| eig.eigenvectors[(r, kept[c])]),
        }
    }

    /// The spectrum with every weight scaled by `s`: the same eigenvectors,
    /// the eigenvalues times `s`.
    pub fn scaled(&self, s: f64) -> Self {
        Self {
            values: &self.values * s,
            vectors: self.vectors.clone(),
        }
    }

    /// The eigenvectors, column `i` times `f(λᵢ)`.
    pub fn weighted_vectors(&self, f: impl Fn(f64) -> f64) -> DMatrix<f64> {
        let mut v = self.vectors.clone();
        for (mut col, &l) in v.column_iter_mut().zip(self.values.iter()) {
            col *= f(l);
        }
        v
    }
}

/// The overlaps `⟨vᵢ, ṽⱼ⟩` (rows: the full graph's modes).
fn overlaps(full: &Spectrum, sparse: &Spectrum) -> DMatrix<f64> {
    full.vectors.transpose() * &sparse.vectors
}

/// Φ between the full graph and the sparse one.
pub(super) fn phi(full: &Spectrum, sparse: &Spectrum) -> f64 {
    phi_of(full, sparse, &overlaps(full, sparse))
}

fn phi_of(full: &Spectrum, sparse: &Spectrum, m: &DMatrix<f64>) -> f64 {
    let mut total = 0.0;
    for (i, &l) in full.values.iter().enumerate() {
        for (j, &lt) in sparse.values.iter().enumerate() {
            total += m[(i, j)].powi(2) * (l - lt).powi(2) / (2.0 * l * lt * (l + lt));
        }
    }
    total
}

/// Φ and what its gradient in each edge weight needs, at one sparse graph.
pub(super) struct Gradient {
    pub phi: f64,
    /// `Uᵀ` and `Xᵀ`: a node's values are a column, read contiguously.
    u: DMatrix<f64>,
    x: DMatrix<f64>,
}

impl Gradient {
    pub fn new(full: &Spectrum, sparse: &Spectrum) -> Self {
        let m = overlaps(full, sparse);
        let k = DMatrix::from_fn(sparse.values.len(), full.values.len(), |j, i| {
            m[(i, j)] / (sparse.values[j] + full.values[i])
        });
        Self {
            phi: phi_of(full, sparse, &m),
            u: (&sparse.vectors * k).transpose(),
            x: sparse.weighted_vectors(f64::recip).transpose(),
        }
    }

    /// `∂Φ/∂w̃` on `edge`: negative where more weight on it lowers Φ.
    pub fn at(&self, edge: &Edge) -> f64 {
        let gap = |m: &DMatrix<f64>| -> f64 {
            m.column(edge.0)
                .iter()
                .zip(m.column(edge.1).iter())
                .map(|(p, q)| (p - q).powi(2))
                .sum()
        };
        2.0 * gap(&self.u) - 0.5 * gap(&self.x)
    }
}
