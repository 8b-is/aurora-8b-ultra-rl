use burn::{
    nn::{Linear, LinearConfig, RmsNorm, RmsNormConfig},
    tensor::{activation, Device, Int, Tensor},
};

/// Single speculative draft block inside the 5-layer `MiMo-V2.6-Pro-RL` Multi-Token Prediction (MTP) head.
pub struct MtpDraftBlock {
    pub norm: RmsNorm,
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
}

impl MtpDraftBlock {
    pub fn init(hidden_dim: usize, device: &Device) -> Self {
        Self {
            norm: RmsNormConfig::new(hidden_dim).with_epsilon(1e-5).init(device),
            gate_proj: LinearConfig::new(hidden_dim, hidden_dim * 2)
                .with_bias(false)
                .init(device),
            up_proj: LinearConfig::new(hidden_dim, hidden_dim * 2)
                .with_bias(false)
                .init(device),
            down_proj: LinearConfig::new(hidden_dim * 2, hidden_dim)
                .with_bias(false)
                .init(device),
        }
    }

    pub fn forward(&self, h: Tensor<3>) -> Tensor<3> {
        let normed = self.norm.forward(h.clone());
        let swiglu = activation::silu(self.gate_proj.forward(normed.clone()))
            * self.up_proj.forward(normed);
        h + self.down_proj.forward(swiglu)
    }
}

/// 5-Layer Multi-Token Prediction (`MTP`) head distilled from `XiaomiMiMo/MiMo-V2.6-Pro-RL`.
/// Drafts `K = 5` future semantic/text tokens per forward pass of the `LFM2.5-8B-A1B` trunk,
/// cutting Discord and ESP32 voice-to-voice Time-To-First-Audio (TTFA) to `< 140 ms`.
pub struct MtpSpeculativeHead {
    pub draft_blocks: Vec<MtpDraftBlock>,
    pub shared_lm_head: Linear,
    pub horizon: usize,
}

impl MtpSpeculativeHead {
    pub fn init(hidden_dim: usize, vocab_size: usize, horizon: usize, device: &Device) -> Self {
        let draft_blocks = (0..horizon)
            .map(|_| MtpDraftBlock::init(hidden_dim, device))
            .collect();
        Self {
            draft_blocks,
            shared_lm_head: LinearConfig::new(hidden_dim, vocab_size)
                .with_bias(false)
                .init(device),
            horizon,
        }
    }

    /// Given trunk hidden state $\mathbf{h}_t$ (`[B, 1, D]`), drafts `horizon` (`5`) future token IDs (`[B, 5]`).
    pub fn draft_future_tokens(&self, mut h_t: Tensor<3>) -> Vec<Tensor<2, Int>> {
        let [b, _, _] = h_t.dims();
        let mut drafted = Vec::with_capacity(self.horizon);
        for block in &self.draft_blocks {
            h_t = block.forward(h_t);
            let logits = self.shared_lm_head.forward(h_t.clone()); // [B, 1, V]
            let v = logits.dims()[2];
            let tok = logits.reshape([b, v]).argmax(1); // [B, 1]
            drafted.push(tok);
        }
        drafted
    }
}
