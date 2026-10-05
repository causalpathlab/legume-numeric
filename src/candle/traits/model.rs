use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

pub trait EncoderModuleT {
    /// An encoder that spits out two results (latent inference, KL loss)
    ///
    /// # Arguments
    /// * `x_nd` - input data (n x d)
    /// * `x0_nd` - null data (n x d)
    /// * `train` - whether to use dropout/batchnorm or not
    ///
    /// # Returns `(z_nk, kl_loss_n)`
    /// * `z_nk` - latent inference (n x k)
    /// * `kl_loss_n` - KL loss (n x 1)
    fn forward_t(
        &self,
        x_nd: &Tensor,
        x0_nd: Option<&Tensor>,
        train: bool,
    ) -> Result<(Tensor, Tensor)>;

    fn dim_latent(&self) -> usize;
}

pub trait JointEncoderModuleT {
    /// An encoder that spits out two results (latent inference, KL loss)
    ///
    /// # Arguments
    /// * `x_nd` - input data (n x d)
    /// * `x0_nd` - null data (n x d)
    /// * `train` - whether to use dropout/batchnorm or not
    ///
    /// # Returns `(z_nk, kl_loss_n)`
    /// * `z_nk` - latent inference (n x k)
    /// * `kl_loss_n` - KL loss (n x 1)
    fn forward_t(
        &self,
        x_nd_vec: &[Tensor],
        x0_nd_vec: &[Option<Tensor>],
        train: bool,
    ) -> Result<(Tensor, Tensor)>;

    fn dim_latent(&self) -> usize;
}

pub trait DecoderModuleT {
    /// A decoder that spits out reconstruction
    fn forward(&self, z_nk: &Tensor) -> Result<Tensor>;

    /// Get a representative dictionary matrix
    fn get_dictionary(&self) -> Result<Tensor>;

    /// A decoder that spits out reconstruction and log-likelihood
    /// * `z_nk` - latent states
    /// * `x_nd` - observed data to validate with
    /// * `llik` - fn (observed, reconstruction) -> log-likelihood
    fn forward_with_llik<LlikFn>(
        &self,
        z_nk: &Tensor,
        x_nd: &Tensor,
        llik: &LlikFn,
    ) -> Result<(Tensor, Tensor)>
    where
        LlikFn: Fn(&Tensor, &Tensor) -> Result<Tensor>;

    /// Whether [`Self::llik_gene_chunked`] actually slices, or falls back to
    /// the dense path. Callers size their memory budget from this, so a
    /// decoder that overrides one must override the other.
    fn llik_is_gene_chunked(&self) -> bool {
        false
    }

    /// Per-cell log-likelihood computed in slices of `gene_chunk` features,
    /// so nothing wider than `[N, gene_chunk]` is ever allocated.
    ///
    /// [`forward_with_llik`](Self::forward_with_llik) must also return the
    /// reconstruction, which is `[N, D]` by construction — training needs it
    /// for the gradient. Inference wants only the scalar per cell, and paying
    /// for a full dense matrix (plus the temporaries behind it) to get one is
    /// what makes whole-transcriptome scoring an OOM.
    ///
    /// The default IS that dense path, so every decoder keeps working
    /// unchanged; a decoder whose rate is separable over genes overrides this
    /// and the memory disappears. Overriding is possible when a gene's rate
    /// does not depend on the other genes' — true of the topic decoders,
    /// whose `logsumexp` reduces over TOPICS so a gene slice is
    /// self-contained, and arranged with a two-pass denominator for the
    /// Gaussian head, whose softmax spans the gene axis.
    fn llik_gene_chunked(&self, z_nk: &Tensor, x_nd: &Tensor, gene_chunk: usize) -> Result<Tensor> {
        let _ = gene_chunk;
        let dense = |_: &Tensor, _: &Tensor| -> Result<Tensor> {
            candle_core::bail!("the dense fallback needs no likelihood closure")
        };
        let (_, llik) = self.forward_with_llik(z_nk, x_nd, &dense)?;
        Ok(llik)
    }

    fn dim_obs(&self) -> usize;

    fn dim_latent(&self) -> usize;

    /// Attach per-feature weights `w_d ∈ (0, 1]` (e.g. NB-Fisher info)
    /// to the decoder loss. Must be at the decoder's `D` resolution.
    /// Default is no-op for decoders that don't support per-feature
    /// reweighting; `MultinomTopicDecoder` overrides.
    fn attach_feature_weights(
        &mut self,
        _weights: &[f32],
        _dev: &candle_core::Device,
    ) -> Result<()> {
        Ok(())
    }
}

/// Shared constructor for topic decoders: `new(n_features, n_topics, VarBuilder)`.
pub trait NewDecoder: Sized {
    fn new(n_features: usize, n_topics: usize, vs: VarBuilder) -> Result<Self>;
}

/// Compute joint multinomial log-likelihood from per-modality log-reconstructions.
/// Returns (recon_vec, total_llik) where total_llik sums across modalities.
pub fn joint_multinomial_llik(
    log_recon_vec: Vec<Tensor>,
    x_nd_vec: &[Tensor],
) -> Result<(Vec<Tensor>, Tensor)> {
    let recon_vec: Vec<Tensor> = log_recon_vec
        .iter()
        .map(|x| x.exp())
        .collect::<Result<Vec<_>>>()?;

    let llik_vec = x_nd_vec
        .iter()
        .zip(&log_recon_vec)
        .map(|(x, log_recon)| -> Result<Tensor> {
            let ret = x
                .clamp(0.0, f64::INFINITY)?
                .mul(log_recon)?
                .sum(x.rank() - 1)?;
            ret.unsqueeze(ret.rank())
        })
        .collect::<Result<Vec<Tensor>>>()?;

    let k = llik_vec[0].rank();
    let llik = Tensor::cat(&llik_vec, k - 1)?.sum(k - 1)?;
    Ok((recon_vec, llik))
}

pub trait JointDecoderModuleT {
    /// A decoder that spits out reconstruction
    fn forward(&self, z_nk: &Tensor) -> Result<Vec<Tensor>>;

    /// Get a representative dictionary matrix
    fn get_dictionary(&self) -> Result<Vec<Tensor>>;

    /// A decoder that spits out reconstruction and log-likelihood
    /// * `z_nk` - latent states
    /// * `x_nd` - observed data to validate with
    /// * `llik` - fn (observed, reconstruction) -> log-likelihood
    fn forward_with_llik<LlikFn>(
        &self,
        z_nk: &Tensor,
        x_nd: &[Tensor],
        llik: &LlikFn,
    ) -> Result<(Vec<Tensor>, Tensor)>
    where
        LlikFn: Fn(&Tensor, &Tensor) -> Result<Tensor>;

    fn dim_obs(&self) -> &[usize];

    fn dim_latent(&self) -> usize;
}
