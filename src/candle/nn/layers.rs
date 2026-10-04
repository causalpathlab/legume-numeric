#![allow(dead_code)]

use candle_core::{Result, Tensor};
use candle_nn::{Activation, Linear, Module};

/// build a stack of alternating `M` and `A` layers
pub struct StackLayers<M>
where
    M: Module,
{
    module_layers: Vec<M>,
    activation_layers: Vec<Option<Activation>>,
}

impl<M> Module for StackLayers<M>
where
    M: Module,
{
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut x = input.clone();
        for (module, activation) in self.module_layers.iter().zip(self.activation_layers.iter()) {
            x = module.forward(&x)?;
            if let Some(activation) = activation {
                x = activation.forward(&x)?;
            }
        }
        Ok(x)
    }
}

impl<M> StackLayers<M>
where
    M: Module,
{
    /// build a stack of alternating `M` and `A` layers
    pub fn new() -> Self {
        Self {
            module_layers: Vec::new(),
            activation_layers: Vec::new(),
        }
    }

    /// add a new layer with an activation function
    pub fn push_with_act(&mut self, layer: M, activation: Activation) {
        self.module_layers.push(layer);
        self.activation_layers.push(Some(activation));
    }

    /// add a new layer
    pub fn push(&mut self, layer: M) {
        self.module_layers.push(layer);
        self.activation_layers.push(None);
    }
}

impl<M> Default for StackLayers<M>
where
    M: Module,
{
    fn default() -> Self {
        Self::new()
    }
}

/// create stacked relu linear
/// * `in_dim` - input
/// * `out_dim` - output
/// * `intermediate_dims` - intermediate layers
/// * `vb` - variable builder
/// * `var_header` - variable name header
pub fn stack_relu_linear(
    in_dim: usize,
    out_dim: usize,
    intermediate_dims: &[usize],
    vb: candle_nn::VarBuilder,
) -> Result<StackLayers<Linear>> {
    let mut prev_dim = in_dim;
    let mut ret = StackLayers::<Linear>::new();
    for (k, &next_dim) in intermediate_dims.iter().enumerate() {
        let _name = format!("relu_linear_stack.{}", k);
        ret.push_with_act(
            candle_nn::linear(prev_dim, next_dim, vb.pp(_name))?,
            candle_nn::Activation::Relu,
        );
        prev_dim = next_dim;
    }
    // add the final layer
    let k = intermediate_dims.len();
    let next_dim = out_dim;
    let _name = format!("relu_linear_stack.{}", k);
    ret.push_with_act(
        candle_nn::linear(prev_dim, next_dim, vb.pp(_name))?,
        candle_nn::Activation::Relu,
    );
    Ok(ret)
}

///////////////////////////////////
// Sparsemax activation function //
///////////////////////////////////

/// Sparsemax activation function (Martins & Astudillo, 2016)
///
/// Projects input onto the probability simplex, producing sparse outputs.
/// Unlike softmax, can output exact zeros.
///
/// * `z` - input tensor of shape (batch, dim)
///
/// Returns tensor of same shape with values in [0, 1] summing to 1 along last dim.
///
pub fn sparsemax(z: &Tensor) -> Result<Tensor> {
    let z = z.contiguous()?; // ensure contiguous for sort_last_dim
    let dim = z.rank() - 1;
    let (z_sorted, _indices) = z.sort_last_dim(false)?; // descending order
    let k = z.dim(dim)?;
    let device = z.device();
    let dtype = z.dtype();

    // Compute cumsum of sorted values
    let cumsum = z_sorted.cumsum(dim)?;

    // Compute 1 + i * z_sorted[i] for i = 1..k
    let range = Tensor::arange(1f32, (k + 1) as f32, device)?.to_dtype(dtype)?;
    // Broadcast range to match z shape
    let shape: Vec<usize> = (0..z.rank())
        .map(|i| if i == dim { k } else { 1 })
        .collect();
    let range = range.reshape(shape.as_slice())?;
    let bound = (z_sorted.broadcast_mul(&range)? + 1.0)?;

    // Find support: where bound > cumsum
    let support = bound.gt(&cumsum)?;

    // Count support size per row (sum of True values)
    let support_f = support.to_dtype(dtype)?;
    let support_size = support_f.sum_keepdim(dim)?;

    // tau = (sum(z_sorted * support) - 1) / support_size
    let z_sorted_masked = z_sorted.broadcast_mul(&support_f)?;
    let z_sum_support = z_sorted_masked.sum_keepdim(dim)?;
    let tau = (z_sum_support - 1.0)?.broadcast_div(&support_size.clamp(1.0, f64::INFINITY)?)?;

    // Output: max(z - tau, 0)
    z.broadcast_sub(&tau)?.clamp(0.0, f64::INFINITY)
}
