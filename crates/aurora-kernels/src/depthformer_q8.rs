use burn::tensor::{activation, Int, Tensor};

/// Fused single-step 8-codebook RVQ Depthformer kernel (`k in 0..8`):
/// 1. Injects residual embedding of previous codebook token $c_{k-1}$ into `state_h` (`[B, D]`)
/// 2. Applies RMSNorm with weight `rms_weight` (`[D]`)
/// 3. Computes fused SwiGLU logits: $\text{SiLU}(z \cdot W_{\text{gate}}) \odot (z \cdot W_{\text{up}})$
/// 4. Returns `(updated_state_h [B, D], logits [B, V], greedy_token [B, 1])`.
pub fn fused_depthformer_q8_step(
    state_h: Tensor<2>,
    prev_code_embed: Option<Tensor<2>>,
    rms_weight: Tensor<1>,
    w_gate: Tensor<2>,
    w_up: Tensor<2>,
    eps: f32,
) -> (Tensor<2>, Tensor<2>, Tensor<2, Int>) {
    let [_b, d] = state_h.dims();
    let z = match prev_code_embed {
        Some(emb) => state_h + emb,
        None => state_h,
    };

    // RMSNorm: z / sqrt(mean(z^2) + eps) * rms_weight
    let ms = z.clone().powf_scalar(2.0).mean_dim(1);
    let inv_rms = (ms + eps).sqrt().recip();
    let z_norm = z.clone() * inv_rms * rms_weight.reshape([1, d]);

    // SwiGLU projection to RVQ codebook logits [B, 2049]
    let gate = activation::silu(z_norm.clone().matmul(w_gate));
    let up = z_norm.matmul(w_up);
    let logits = gate * up;
    let tok = logits.clone().argmax(1);

    (z, logits, tok)
}
