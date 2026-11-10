use super::super::tests::{fuzzy_edges, spacing, square_graph};
use super::*;
use crate::matrix::pca::{init_2d_from_scores, pc_scores};
use crate::matrix::traits::MatOps;
use nalgebra::DMatrix;
use rand::{rngs::SmallRng, RngExt, SeedableRng};

/// Epochs `from..umap.n_epochs` on `dev`, from `y`.
fn epochs_on(
    umap: &Umap,
    edges: &[(usize, usize, f32)],
    y: &[f32],
    from: usize,
    dev: &Device,
) -> Vec<f32> {
    let n = y.len() / 2;
    let graph = Graph::new(edges, n, dev).unwrap();
    let y = Tensor::from_slice(y, (n, 2), dev).unwrap();
    let y = umap.run_epochs(&graph, y, from).unwrap();
    y.flatten_all().unwrap().to_vec1().unwrap()
}

#[test]
fn an_edge_is_sampled_as_often_as_the_lock_free_schedule_says() {
    // Weights 1/2^k keep the reference's running sums exact, so the two
    // must agree epoch by epoch.
    let weights = [1.0_f32, 0.5, 0.25, 0.125, 0.0];
    let mut next: Vec<f32> = weights
        .iter()
        .map(|&w| if w > 0.0 { 1.0 / w } else { f32::INFINITY })
        .collect();
    for epoch in 0..64 {
        for (e, ne) in next.iter_mut().enumerate() {
            let want = *ne <= epoch as f32;
            if want {
                *ne += 1.0 / weights[e];
            }
            assert_eq!(
                sampled(weights[e], epoch as f32),
                want,
                "edge {e} epoch {epoch}"
            );
        }
    }
}

#[test]
fn negative_samples_cover_the_points_evenly() {
    let p = Params {
        n: 37,
        n_neg: 5,
        epoch: 3,
        key: epoch_key(42, 3),
        alpha: 1.0,
        a: 1.0,
        b: 1.0,
    };
    let mut counts = vec![0_usize; p.n as usize];
    for d in 0..20_000 {
        for s in 0..p.n_neg {
            counts[negative(&p, d, s) as usize] += 1;
        }
    }
    let mean = 100_000.0 / p.n as f32;
    for (r, &c) in counts.iter().enumerate() {
        assert!(
            (c as f32 - mean).abs() < 0.1 * mean,
            "point {r}: {c} draws, mean {mean}"
        );
    }
}

/// t-UMAP, d^2 = 1, alpha = 1/4: each end moves 2 * alpha * 2/(1+1) * 1
/// toward the other, once as head and once as the other's tail.
fn check_one_edge_on(dev: &Device) {
    let umap = Umap {
        n_epochs: 2,
        learning_rate: 0.5,
        negative_sample_rate: 0,
        ..Umap::tumap()
    };
    let got = epochs_on(&umap, &[(0, 1, 1.0)], &[0.0, 0.0, 1.0, 0.0], 1, dev);
    let want = [0.5, 0.0, 0.5, 0.0];
    assert!(
        got.iter().zip(&want).all(|(a, b)| (a - b).abs() < 1e-6),
        "{got:?}"
    );
}

#[test]
fn one_edge_pulls_its_two_ends_together_by_the_kernel_gradient() {
    check_one_edge_on(&Device::Cpu);
}

/// Uniform points in a square, laid out by t-UMAP on `dev`.
fn square_layout_on(n: usize, dev: &Device) -> Vec<f32> {
    let (edges, init) = square_graph(n);
    Umap {
        n_epochs: 200,
        ..Umap::tumap()
    }
    .fit_on(&edges, n, &init, dev)
    .unwrap()
}

fn check_spaced_and_centred_on(dev: &Device) {
    let n = 1500;
    let y = square_layout_on(n, dev);
    let ratio = spacing(&y, n, n / 2..n) / spacing(&y, n, 0..n / 2);
    assert!((0.9..1.1).contains(&ratio), "high/low spacing {ratio}");
    for c in 0..2 {
        let mean = (0..n).map(|i| y[i * 2 + c]).sum::<f32>() / n as f32;
        assert!(mean.abs() < 1e-3, "mean {mean}");
    }
}

#[test]
fn points_are_spaced_alike_and_centred() {
    check_spaced_and_centred_on(&Device::Cpu);
}

/// Mean fraction of each query's `k` nearest neighbours in `data` that
/// stay among its `k` nearest in the layout `y`.
fn neighbours_kept(data: &DMatrix<f32>, y: &[f32], k: usize, queries: usize) -> f32 {
    let n = data.nrows();
    let near = |dist: &dyn Fn(usize) -> f32, i: usize| {
        let mut v: Vec<(f32, usize)> = (0..n).filter(|&j| j != i).map(|j| (dist(j), j)).collect();
        v.select_nth_unstable_by(k, |a, b| a.0.total_cmp(&b.0));
        v.truncate(k);
        v.into_iter().map(|x| x.1).collect::<Vec<_>>()
    };
    let qs: Vec<usize> = (0..n).step_by((n / queries).max(1)).collect();
    let hits: usize = qs
        .iter()
        .map(|&i| {
            let hi = near(&|j| (data.row(i) - data.row(j)).norm_squared(), i);
            let lo = near(
                &|j| (y[i * 2] - y[j * 2]).powi(2) + (y[i * 2 + 1] - y[j * 2 + 1]).powi(2),
                i,
            );
            lo.iter().filter(|j| hi.contains(j)).count()
        })
        .sum();
    hits as f32 / (qs.len() * k) as f32
}

/// A 2-d square embedded in 20 dimensions with a little noise: a layout
/// can keep most neighbours, so the measure tells fits apart.
fn embedded_square(n: usize, seed: u64) -> DMatrix<f32> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let basis = DMatrix::from_fn(2, 20, |_, _| rng.random_range(-1.0_f32..1.0));
    let square = DMatrix::from_fn(n, 2, |_, _| rng.random_range(0.0_f32..10.0));
    let noise = DMatrix::from_fn(n, 20, |_, _| rng.random_range(-0.05_f32..0.05));
    square * basis + noise
}

/// The PCA start layouts are fitted from in practice.
fn pca_start(data: &DMatrix<f32>) -> Vec<f32> {
    init_2d_from_scores(&pc_scores(&data.centre_columns(), 2, 0).unwrap(), 6)
}

/// From a PCA start the synchronous epochs keep neighbours as well as the
/// lock-free CPU epochs do.
fn check_neighbours_kept_on(dev: &Device) {
    let n = 2000;
    let data = embedded_square(n, 5);
    let edges = fuzzy_edges(&data, 15);
    let init = pca_start(&data);
    let umap = Umap {
        n_epochs: 300,
        ..Umap::default()
    };
    let lock_free = neighbours_kept(&data, &umap.fit(&edges, n, &init), 15, 400);
    let synchronous = neighbours_kept(&data, &umap.fit_on(&edges, n, &init, dev).unwrap(), 15, 400);
    assert!(
        synchronous >= lock_free - 0.03,
        "kept {synchronous} vs lock-free {lock_free}"
    );
}

#[test]
fn neighbours_are_kept_as_well_as_by_the_lock_free_fit() {
    check_neighbours_kept_on(&Device::Cpu);
}

#[cfg(any(feature = "metal", feature = "cuda"))]
mod gpu {
    use super::*;

    /// Each of the first epochs on `dev` against the CPU op from the same
    /// coordinates, for t-UMAP and the standard kernel, on a random graph with
    /// a hub. Epoch by epoch, as SGD at rate 1 amplifies rounding about tenfold
    /// an epoch.
    fn check_matches_cpu_on(dev: &Device) {
        let mut rng = SmallRng::seed_from_u64(7);
        let n = 200;
        let mut edges: Vec<(usize, usize, f32)> = (0..1500)
            .map(|_| {
                let i = rng.random_range(0..n);
                let j = (i + rng.random_range(1..n)) % n;
                (i.min(j), i.max(j), rng.random_range(0.05..1.0))
            })
            .collect();
        edges.extend((1..120).map(|j| (0, j, 0.7)));
        let y: Vec<f32> = (0..n * 2).map(|_| rng.random_range(-3.0..3.0)).collect();
        for (a, b) in [(1.0, 1.0), (1.929, 0.7915)] {
            let mut y = y.clone();
            for epoch in 1..6 {
                let umap = Umap {
                    n_epochs: epoch + 1,
                    negative_sample_rate: 3,
                    a,
                    b,
                    ..Umap::default()
                };
                let want = epochs_on(&umap, &edges, &y, epoch, &Device::Cpu);
                let got = epochs_on(&umap, &edges, &y, epoch, dev);
                let worst = got
                    .iter()
                    .zip(&want)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f32::max);
                assert!(
                    worst < 1e-5,
                    "(a, b) = ({a}, {b}) epoch {epoch}: worst {worst}"
                );
                y = want;
            }
        }
    }

    /// The GPU this build targets: CUDA when built with it, else Metal.
    fn gpu() -> Device {
        if cfg!(feature = "cuda") {
            Device::new_cuda(0).unwrap()
        } else {
            Device::new_metal(0).unwrap()
        }
    }

    #[test]
    fn on_the_gpu_one_edge_pulls_its_ends_together() {
        check_one_edge_on(&gpu());
    }

    #[test]
    fn on_the_gpu_epochs_match_the_cpu() {
        check_matches_cpu_on(&gpu());
    }

    #[test]
    fn on_the_gpu_points_are_spaced_alike_and_centred() {
        check_spaced_and_centred_on(&gpu());
    }

    #[test]
    fn on_the_gpu_neighbours_are_kept_as_well_as_by_the_lock_free_fit() {
        check_neighbours_kept_on(&gpu());
    }

    /// Timing against the lock-free fit, on a square embedded in 20
    /// dimensions, from its PCA. Run with
    /// `cargo test --release --features metal --lib umap_bench -- --ignored --nocapture`
    /// (or `--features cuda`); `UMAP_BENCH_N` sets the size.
    #[test]
    #[ignore]
    fn umap_bench_gpu_against_lock_free() {
        use std::time::Instant;
        let dev = gpu();
        let n: usize = std::env::var("UMAP_BENCH_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(100_000);
        let data = embedded_square(n, 9);
        let edges = fuzzy_edges(&data, 15);
        let init = pca_start(&data);
        let umap = Umap::default();
        // Compile the kernel (once per device) before timing.
        Umap {
            n_epochs: 3,
            ..Umap::default()
        }
        .fit_on(&edges, n, &init, &dev)
        .unwrap();

        let t = Instant::now();
        let lock_free = umap.fit(&edges, n, &init);
        let t_lock_free = t.elapsed();
        let t = Instant::now();
        let on_cpu = umap.fit_on(&edges, n, &init, &Device::Cpu).unwrap();
        let t_cpu = t.elapsed();
        let t = Instant::now();
        let on_gpu = umap.fit_on(&edges, n, &init, &dev).unwrap();
        let t_gpu = t.elapsed();
        let kept = |y: &[f32]| neighbours_kept(&data, y, 15, 500);
        println!(
            "n {n}, {} edges, {:?}\n  lock-free  {t_lock_free:>10.2?}  kept {:.3}\n  sync cpu   {t_cpu:>10.2?}  kept {:.3}\n  sync gpu   {t_gpu:>10.2?}  kept {:.3}\n  gpu speedup over lock-free {:.2}",
            edges.len(),
            dev.location(),
            kept(&lock_free),
            kept(&on_cpu),
            kept(&on_gpu),
            t_lock_free.as_secs_f64() / t_gpu.as_secs_f64()
        );
    }
}
