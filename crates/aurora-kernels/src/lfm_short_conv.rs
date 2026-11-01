use burn::tensor::{activation, module::conv1d, ops::ConvOptions, Tensor};

/// Fused LFM2.5 double-gated short causal convolution step (`K = 3`):
/// 1. Unpacks `[B_t, C_t, x_t]` from `proj_bcx` (`[B, T, 3 * D]`)
/// 2. Applies input SiLU gate: $v_t = x_t \odot \text{SiLU}(B_t)$
/// 3. Prepends rolling `conv_cache` (`[B, D, K - 1]`) and executes depthwise causal `conv1d`
/// 4. Updates `conv_cache` with the trailing `K - 1` states for the next `80 ms` tick
/// 5. Applies output SiLU gate: $y_t = u_t \odot \text{SiLU}(C_t)$
pub fn fused_lfm_short_conv_step(
    proj_bcx: Tensor<3>,
    conv_weight: Tensor<3>,
    conv_cache: Tensor<3>,
) -> (Tensor<3>, Tensor<3>) {
    let [b, _t, d3] = proj_bcx.dims();
    let d = d3 / 3;
    let [_, _, k] = conv_weight.dims();
    let cache_len = k.saturating_sub(1);

    let chunks = proj_bcx.chunk(3, 2);
    let (b_gate, c_gate, x_val) = (chunks[0].clone(), chunks[1].clone(), chunks[2].clone());

    // First input gate: v = x * SiLU(B), transposed to [B, D, T]
    let v_gated = (x_val * activation::silu(b_gate)).swap_dims(1, 2);

    // Prepend rolling [B, D, K-1] state cache
    let padded = Tensor::cat(vec![conv_cache, v_gated], 2);
    let total_t = padded.dims()[2];
    let next_cache = padded
        .clone()
        .slice([0..b, 0..d, (total_t - cache_len)..total_t]);

    // Depthwise causal 1D convolution (groups = D, padding = 0)
    let u_conv = conv1d(
        padded,
        conv_weight,
        None,
        ConvOptions::new([1], [0], [1], d),
    )
    .swap_dims(1, 2);

    // Second output gate: y = u * SiLU(C)
    let y_out = u_conv * activation::silu(c_gate);
    (y_out, next_cache)
}
