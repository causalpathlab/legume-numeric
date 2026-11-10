//! UMAP SGD on a candle device: CPU, Metal or CUDA.
//!
//! The lock-free CPU fit updates coordinates one edge at a time. A GPU
//! runs every point at once, so here each epoch is synchronous, as cuML's
//! deterministic mode: every gradient is taken from the coordinates at the
//! start of the epoch, and each point moves by the sum of its gradients.
//! The epoch schedule, kernel, clamping and negative sampling rate are
//! those of [`Umap::fit`].
//!
//! Every update to a point comes from a directed edge it heads: an edge
//! `(i, j)` pulls `i` toward `j` as its head and again as `j`'s tail, and
//! its negative samples push only the head. So one thread per point walks
//! the edges it heads and writes only its own row: no atomics, and the
//! result does not depend on scheduling. An epoch is one candle
//! [`InplaceOp1`]: plain Rust on CPU, and on Metal and CUDA one kernel
//! compiled at first use from `device/epoch.h`, which mirrors [`point`].
//! Negative samples come from a hash of the seed, epoch, edge and draw,
//! so every device draws the same ones.

use super::{attraction_coeff, clamp4, repulsion_coeff, Umap};
use candle_core::{CpuStorage, Device, InplaceOp1, Layout, Result, Storage, Tensor};
use rayon::prelude::*;
use std::sync::RwLockReadGuard;

/// Kernel arguments for one epoch: `UmapParams` in `device/epoch.h`.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct Params {
    n: u32,
    n_neg: u32,
    epoch: u32,
    /// The seed and epoch, hashed: [`negative`] mixes in the edge and draw.
    key: u32,
    alpha: f32,
    a: f32,
    b: f32,
}

// SAFETY: `Params` is `#[repr(C)]` plain data, passed by value as the
// kernel's `UmapParams`.
#[cfg(feature = "cuda")]
unsafe impl candle_core::cuda_backend::cudarc::driver::DeviceRepr for Params {}

/// The edge list as CSR over heads: directed edges `row_ptr[i]..row_ptr[i + 1]`
/// have head `i`, tail `tails[d]` and relative weight `q[d] = w / max w`.
struct Graph {
    row_ptr: Tensor,
    tails: Tensor,
    q: Tensor,
}

impl Graph {
    fn new(edges: &[(usize, usize, f32)], n: usize, dev: &Device) -> Result<Self> {
        let max_w = edges.iter().map(|e| e.2).fold(0.0_f32, f32::max).max(1e-4);
        let mut row_ptr = vec![0_u32; n + 1];
        for &(i, j, _) in edges {
            row_ptr[i + 1] += 1;
            row_ptr[j + 1] += 1;
        }
        for i in 0..n {
            row_ptr[i + 1] += row_ptr[i];
        }
        let m = edges.len() * 2;
        let (mut tails, mut q) = (vec![0_u32; m], vec![0.0_f32; m]);
        let mut next: Vec<u32> = row_ptr[..n].to_vec();
        for &(i, j, w) in edges {
            let r = w.max(0.0) / max_w;
            for (h, t) in [(i, j), (j, i)] {
                let d = next[h] as usize;
                (tails[d], q[d]) = (t as u32, r);
                next[h] += 1;
            }
        }
        Ok(Self {
            row_ptr: Tensor::from_vec(row_ptr, n + 1, dev)?,
            tails: Tensor::from_vec(tails, m, dev)?,
            q: Tensor::from_vec(q, m, dev)?,
        })
    }
}

/// One epoch: writes the tensor it runs on from `y_in`.
struct Epoch<'a> {
    y_in: &'a Tensor,
    graph: &'a Graph,
    p: Params,
}

/// A contiguous tensor's storage, read-locked.
fn storage(t: &Tensor) -> Result<RwLockReadGuard<'_, Storage>> {
    let (s, l) = t.storage_and_layout();
    if l.contiguous_offsets() != Some((0, t.elem_count())) {
        candle_core::bail!("umap epoch needs whole contiguous tensors");
    }
    Ok(s)
}

impl Epoch<'_> {
    /// The storages of `y_in`, `row_ptr`, `tails` and `q`, read-locked.
    fn inputs(&self) -> Result<[RwLockReadGuard<'_, Storage>; 4]> {
        let g = self.graph;
        Ok([
            storage(self.y_in)?,
            storage(&g.row_ptr)?,
            storage(&g.tails)?,
            storage(&g.q)?,
        ])
    }
}

fn hash(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^ (x >> 16)
}

/// [`Params::key`] for `epoch` of a fit seeded with `seed`.
fn epoch_key(seed: u64, epoch: u32) -> u32 {
    let h = hash(epoch.wrapping_mul(0x9e37_79b9));
    hash(hash(h ^ seed as u32) ^ (seed >> 32) as u32)
}

/// Negative sample `s` of directed edge `d`, uniform on `[0, n)`.
fn negative(p: &Params, d: u32, s: u32) -> u32 {
    let h = hash(hash(d ^ p.key) ^ s);
    ((h as u64 * p.n as u64) >> 32) as u32
}

/// Whether an edge of relative weight `q` is sampled at `epoch`: when
/// `floor(epoch * q)` steps up, as [`Umap::fit`]'s schedule of every
/// `1 / q` epochs, which samples nothing at epoch 0.
fn sampled(q: f32, epoch: f32) -> bool {
    epoch > 0.0 && (epoch * q).floor() > ((epoch - 1.0) * q).floor()
}

/// Point `i`'s next coordinates: `umap_point` in `device/epoch.h`.
fn point(i: usize, y: &[f32], row_ptr: &[u32], tails: &[u32], q: &[f32], p: &Params) -> [f32; 2] {
    let (a, b, alpha) = (p.a, p.b, p.alpha);
    let (xi, yi) = (y[2 * i], y[2 * i + 1]);
    let (mut gx, mut gy) = (0.0_f32, 0.0_f32);
    let e = p.epoch as f32;
    for d in row_ptr[i]..row_ptr[i + 1] {
        if !sampled(q[d as usize], e) {
            continue;
        }
        let t = tails[d as usize] as usize;
        let (dx, dy) = (xi - y[2 * t], yi - y[2 * t + 1]);
        let d2 = dx * dx + dy * dy;
        if d2 > 0.0 {
            let c = attraction_coeff(d2, a, b);
            gx += clamp4(dx * c) * (2.0 * alpha);
            gy += clamp4(dy * c) * (2.0 * alpha);
        }
        for s in 0..p.n_neg {
            let r = negative(p, d, s) as usize;
            let (rx, ry) = (xi - y[2 * r], yi - y[2 * r + 1]);
            let r2 = rx * rx + ry * ry;
            if r2 > 0.0 {
                let c = repulsion_coeff(r2, a, b);
                gx += clamp4(rx * c) * alpha;
                gy += clamp4(ry * c) * alpha;
            }
        }
    }
    [xi + gx, yi + gy]
}

#[cfg(any(feature = "metal", feature = "cuda"))]
const EPOCH_H: &str = include_str!("device/epoch.h");

#[cfg(feature = "metal")]
const METAL_PRELUDE: &str = "#include <metal_stdlib>
using namespace metal;
#define FN inline
#define DEV device
#define POW(x, y) precise::pow(x, y)
#define FLOOR(x) floor(x)
#define FMIN(x, y) fmin(x, y)
#define FMAX(x, y) fmax(x, y)
#define MULHI(x, y) mulhi(x, y)
";

#[cfg(feature = "metal")]
const METAL_KERNEL: &str = "
kernel void umap_epoch(device const float* y_in [[buffer(0)]], device float* y_out [[buffer(1)]],
                       device const u32* row_ptr [[buffer(2)]], device const u32* tails [[buffer(3)]],
                       device const float* q [[buffer(4)]], constant UmapParams& p [[buffer(5)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    umap_point(i, y_in, y_out, row_ptr, tails, q, p);
}
";

#[cfg(feature = "cuda")]
const CUDA_PRELUDE: &str = "#define FN __device__ __forceinline__
#define DEV
#define POW(x, y) powf(x, y)
#define FLOOR(x) floorf(x)
#define FMIN(x, y) fminf(x, y)
#define FMAX(x, y) fmaxf(x, y)
#define MULHI(x, y) __umulhi(x, y)
";

#[cfg(feature = "cuda")]
const CUDA_KERNEL: &str = "
extern \"C\" __global__ void umap_epoch(const float* y_in, float* y_out, const u32* row_ptr,
                                       const u32* tails, const float* q, UmapParams p) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.n) return;
    umap_point(i, y_in, y_out, row_ptr, tails, q, p);
}
";

/// The epoch pipeline for `md`, compiled once per device. candle caches
/// only its own Metal kernels, so this keeps the compiled source.
#[cfg(feature = "metal")]
fn metal_pipeline(
    md: &candle_core::MetalDevice,
) -> Result<candle_metal_kernels::metal::ComputePipeline> {
    use candle_core::metal_backend::DeviceId;
    use candle_core::MetalError;
    use candle_metal_kernels::metal::ComputePipeline;
    static COMPILED: std::sync::Mutex<Vec<(DeviceId, ComputePipeline)>> =
        std::sync::Mutex::new(Vec::new());
    let mut compiled = COMPILED.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, p)) = compiled.iter().find(|(id, _)| *id == md.id()) {
        return Ok(p.clone());
    }
    let src = format!("{METAL_PRELUDE}{EPOCH_H}{METAL_KERNEL}");
    let lib = md
        .device()
        .new_library_with_source(&src, None)
        .map_err(MetalError::from)?;
    let f = lib
        .get_function("umap_epoch", None)
        .map_err(MetalError::from)?;
    let p = md
        .device()
        .new_compute_pipeline_state_with_function(&f)
        .map_err(MetalError::from)?;
    compiled.push((md.id(), p.clone()));
    Ok(p)
}

/// The kernel's PTX, compiled once per process through nvrtc; candle
/// loads it once per device.
#[cfg(feature = "cuda")]
fn cuda_ptx() -> Result<&'static str> {
    use std::sync::OnceLock;
    static PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();
    match PTX.get_or_init(|| {
        candle_core::cuda_backend::cudarc::nvrtc::compile_ptx(format!(
            "{CUDA_PRELUDE}{EPOCH_H}{CUDA_KERNEL}"
        ))
        .map(|p| p.to_src())
        .map_err(|e| format!("{e:?}"))
    }) {
        Ok(ptx) => Ok(ptx.as_str()),
        Err(e) => Err(candle_core::Error::Msg(format!(
            "umap epoch: nvrtc failed: {e}"
        ))),
    }
}

impl InplaceOp1 for Epoch<'_> {
    fn name(&self) -> &'static str {
        "umap-epoch"
    }

    fn cpu_fwd(&self, out: &mut CpuStorage, _: &Layout) -> Result<()> {
        let CpuStorage::F32(out) = out else {
            candle_core::bail!("umap epoch needs f32 coordinates")
        };
        let [y, r, t, q] = self.inputs()?;
        let (
            Storage::Cpu(CpuStorage::F32(y)),
            Storage::Cpu(CpuStorage::U32(r)),
            Storage::Cpu(CpuStorage::U32(t)),
            Storage::Cpu(CpuStorage::F32(q)),
        ) = (&*y, &*r, &*t, &*q)
        else {
            candle_core::bail!("umap epoch inputs must be on the cpu")
        };
        out.par_chunks_mut(2)
            .enumerate()
            .for_each(|(i, o)| o.copy_from_slice(&point(i, y, r, t, q, &self.p)));
        Ok(())
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(&self, out: &mut candle_core::MetalStorage, _: &Layout) -> Result<()> {
        use candle_core::backend::BackendStorage;
        use objc2_metal::MTLSize;
        let md = out.device().clone();
        let pipeline = metal_pipeline(&md)?;
        let [y, r, t, q] = self.inputs()?;
        let (Storage::Metal(y), Storage::Metal(r), Storage::Metal(t), Storage::Metal(q)) =
            (&*y, &*r, &*t, &*q)
        else {
            candle_core::bail!("umap epoch inputs must be on the metal device")
        };
        let n = self.p.n as usize;
        let width = pipeline.max_total_threads_per_threadgroup().min(n.max(1));
        let enc = md.command_encoder()?;
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(y.buffer()), 0);
        enc.set_buffer(1, Some(out.buffer()), 0);
        enc.set_buffer(2, Some(r.buffer()), 0);
        enc.set_buffer(3, Some(t.buffer()), 0);
        enc.set_buffer(4, Some(q.buffer()), 0);
        enc.set_bytes(5, &self.p);
        enc.dispatch_thread_groups(
            MTLSize {
                width: n.div_ceil(width),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, out: &mut candle_core::CudaStorage, _: &Layout) -> Result<()> {
        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;
        /// Threads per block: a 1024-thread block caps occupancy on
        /// Ampere's 1536 threads per SM, and rows vary in length.
        const BLOCK: u32 = 256;
        let func =
            out.device()
                .get_or_load_custom_func("umap_epoch", "legume_umap_epoch", cuda_ptx()?)?;
        let [y, r, t, q] = self.inputs()?;
        let (Storage::Cuda(y), Storage::Cuda(r), Storage::Cuda(t), Storage::Cuda(q)) =
            (&*y, &*r, &*t, &*q)
        else {
            candle_core::bail!("umap epoch inputs must be on the cuda device")
        };
        let (y, r, t, q) = (
            y.as_cuda_slice::<f32>()?,
            r.as_cuda_slice::<u32>()?,
            t.as_cuda_slice::<u32>()?,
            q.as_cuda_slice::<f32>()?,
        );
        let out = out.as_cuda_slice_mut::<f32>()?;
        let cfg = LaunchConfig {
            grid_dim: (self.p.n.div_ceil(BLOCK), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = func.builder();
        b.arg(y).arg(out).arg(r).arg(t).arg(q).arg(&self.p);
        // SAFETY: the arguments match `umap_epoch` in CUDA_KERNEL, and the
        // kernel writes only `out[0..2n]`.
        unsafe { b.launch(cfg) }.w()?;
        Ok(())
    }
}

impl Umap {
    /// Epochs `from..` of the fit, from `y` (`n x 2` on `dev`).
    fn run_epochs(&self, graph: &Graph, y: Tensor, from: usize) -> Result<Tensor> {
        let n = y.dim(0)?;
        let (mut y_in, mut y_out) = (y.clone(), y.zeros_like()?);
        for epoch in from..self.n_epochs {
            let p = Params {
                n: n as u32,
                n_neg: self.negative_sample_rate as u32,
                epoch: epoch as u32,
                key: epoch_key(self.seed, epoch as u32),
                alpha: self.learning_rate * (1.0 - epoch as f32 / self.n_epochs as f32),
                a: self.a,
                b: self.b,
            };
            y_out.inplace_op1(&Epoch {
                y_in: &y_in,
                graph,
                p,
            })?;
            std::mem::swap(&mut y_in, &mut y_out);
        }
        Ok(y_in)
    }

    /// [`Umap::fit`] on a candle device, for Metal or CUDA on large graphs.
    ///
    /// The epochs are synchronous (see the module notes), so the layout
    /// differs from [`Umap::fit`]'s and is the same on every device up to
    /// rounding. Start it from PCA: from there it keeps neighbours as well
    /// as [`Umap::fit`], while from a random start it keeps fewer, as each
    /// point moves by all its pulls at once. On CPU, [`Umap::fit`] is as
    /// fast; this path is there for layouts that match across devices.
    /// Returns row-major `n x 2`, centred.
    pub fn fit_on(
        &self,
        edges: &[(usize, usize, f32)],
        n: usize,
        init: &[f32],
        dev: &Device,
    ) -> anyhow::Result<Vec<f32>> {
        anyhow::ensure!(init.len() == n * 2, "init size mismatch");
        anyhow::ensure!(
            u32::try_from(n).is_ok() && u32::try_from(edges.len() * 2).is_ok(),
            "fit_on indexes points and directed edges in u32: {n} points, {} edges",
            edges.len()
        );
        anyhow::ensure!(
            edges.iter().all(|&(i, j, _)| i < n && j < n),
            "an edge endpoint is not below n = {n}"
        );
        if n == 0 {
            return Ok(Vec::new());
        }
        let graph = Graph::new(edges, n, dev)?;
        let y = Tensor::from_slice(init, (n, 2), dev)?;
        // Epoch 0 samples no edge, as in `fit`.
        let y = self.run_epochs(&graph, y, 1)?;
        let y = y.broadcast_sub(&y.mean_keepdim(0)?)?;
        Ok(y.flatten_all()?.to_vec1()?)
    }
}

#[cfg(test)]
mod tests;
