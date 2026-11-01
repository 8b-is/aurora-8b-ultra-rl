use burn::tensor::{activation, module::conv1d, ops::ConvOptions, Tensor};

/// Derivative of $\text{SiLU}(z) = z \cdot \sigma(z)$:
/// $$\text{SiLU}'(z) = \sigma(z) \cdot (1 + z \cdot (1 - \sigma(z)))$$
#[inline]
pub fn silu_backward(z: Tensor<3>) -> Tensor<3> {
    let sig = activation::sigmoid(z.clone());
    sig.clone() * (Tensor::ones(z.dims(), &z.device()) + z * (Tensor::ones(sig.dims(), &sig.device()) - sig))
}

/// Analytical backward pass for the fused LFM2.5 double-gated short causal convolution kernel
/// used during Stage 2 SFT and Stage 4 GRPO reinforcement learning.
///
/// Given upstream `grad_y` (`[B, T, D]`), saved `proj_bcx` (`[B, T, 3*D]`), `conv_weight` (`[D, 1, K]`),
/// and `conv_cache` (`[B, D, K-1]`), computes `(grad_proj_bcx [B, T, 3*D], grad_conv_weight [D, 1, K])`.
pub fn fused_lfm_short_conv_backward(
    grad_y: Tensor<3>,
    proj_bcx: Tensor<3>,
    conv_weight: Tensor<3>,
    conv_cache: Tensor<3>,
) -> (Tensor<3>, Tensor<3>) {
    let [b, t, d3] = proj_bcx.dims();
    let d = d3 / 3;
    let [_, _, k] = conv_weight.dims();

    let chunks = proj_bcx.chunk(3, 2);
    let (b_gate, c_gate, x_val) = (chunks[0].clone(), chunks[1].clone(), chunks[2].clone());

    let silu_b = activation::silu(b_gate.clone());
    let silu_c = activation::silu(c_gate.clone());
    let dsilu_b = silu_backward(b_gate);
    let dsilu_c = silu_backward(c_gate);

    // Forward intermediate v = x * SiLU(B) -> [B, D, T]
    let v_gated = (x_val.clone() * silu_b.clone()).swap_dims(1, 2);
    let padded = Tensor::cat(vec![conv_cache, v_gated], 2); // [B, D, (K-1) + T]

    // Forward causal depthwise conv u -> [B, T, D]
    let u_conv = conv1d(
        padded.clone(),
        conv_weight.clone(),
        None,
        ConvOptions::new([1], [0], [1], d),
    )
    .swap_dims(1, 2);

    // 1. Gradient through C-gate: y = u * SiLU(C)
    let grad_c = grad_y.clone() * u_conv * dsilu_c;
    let grad_u = (grad_y * silu_c).swap_dims(1, 2); // [B, D, T]

    // 2. Gradient w.r.t. conv_weight [D, 1, K] and v_gated [B, D, T]
    let mut grad_w_tap_slices = Vec::with_capacity(k);
    let mut grad_v = Tensor::<3>::zeros([b, d, t], &grad_u.device());

    for tap in 0..k {
        let v_window = padded.clone().slice([0..b, 0..d, tap..(tap + t)]);
        // Sum over batch and time: [D, 1, 1]
        let dw_tap = (grad_u.clone() * v_window)
            .sum_dim(0)
            .sum_dim(2)
            .reshape([d, 1, 1]);
        grad_w_tap_slices.push(dw_tap);

        // Propagate grad_u back to v_gated for causal tap positions
        let w_tap = conv_weight
            .clone()
            .slice([0..d, 0..1, tap..(tap + 1)])
            .reshape([1, d, 1]);
        let shift = (k - 1) - tap;
        if shift == 0 {
            grad_v = grad_v + grad_u.clone() * w_tap;
        } else if shift < t {
            let gu_sub = grad_u.clone().slice([0..b, 0..d, shift..t]);
            let pad_tail = Tensor::<3>::zeros([b, d, shift], &grad_u.device());
            let shifted = Tensor::cat(vec![gu_sub, pad_tail], 2);
            grad_v = grad_v + shifted * w_tap;
        }
    }

    let grad_conv_weight = Tensor::cat(grad_w_tap_slices, 2); // [D, 1, K]
    let grad_v_btd = grad_v.swap_dims(1, 2); // [B, T, D]

    // 3. Gradient through B-gate and x: v = x * SiLU(B)
    let grad_x = grad_v_btd.clone() * silu_b;
    let grad_b = grad_v_btd * x_val * dsilu_b;

    let grad_proj_bcx = Tensor::cat(vec![grad_b, grad_c, grad_x], 2);
    (grad_proj_bcx, grad_conv_weight)
}
