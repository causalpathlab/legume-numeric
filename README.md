# legume-numeric

Numeric and ML foundation for the [legume](https://github.com/causalpathlab/legume-rs) ecosystem.

## Modules

| Module | Feature | Role |
|--------|---------|------|
| `leiden` | (always) | Vendored CWTS/10x Leiden port — see `NOTICE` |
| `matrix` | `matrix` (default) | Matrix IO, KNN, layout, stats |
| `param` | `param` (default) | Parametric / model helpers |
| `candle` | `candle` | candle-core/nn training helpers |
| `mcmc` | `mcmc` | MCMC engine + sparse regression |

## Features

```toml
legume-numeric = { version = "0.8", features = ["candle", "mcmc"] }
# GPU (optional):
# features = ["candle", "mcmc", "cuda"]  # or "metal"
```

Default features are `matrix` + `param` (CPU). Uses crates.io `candle-core` / `candle-nn` 0.10.

## License

MIT. The `leiden` module includes third-party copyright (CWTS Leiden University / 10x Genomics); see `NOTICE`.
