use burn::{
    module::Param,
    nn::{
        conv::{Conv1d, Conv1dConfig},
        Embedding, EmbeddingConfig, Linear, LinearConfig, PaddingConfig1d, RmsNorm, RmsNormConfig,
    },
    tensor::{activation, Device, Int, Tensor},
};
use serde::{Deserialize, Serialize};

/// Architecture configuration for `LiquidAI/LFM2.5-8B-A1B` (`lfm2_moe`) extended with
/// `Aurora-8B-Ultra-RL` quantal $p/q$ vocabulary (`65,536` text + `8,192` semantic audio + `16` VAD/control = `73,744`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lfm2MoeConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_dense_layers: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub conv_l_cache: usize,
    pub norm_eps: f64,
    pub routed_scaling_factor: f32,
    pub full_attention_layers: Vec<usize>,
}

impl Default for Lfm2MoeConfig {
    /// Default full `LiquidAI/LFM2.5-8B-A1B` (`8.3B` total / `1.5B` active) configuration.
    fn default() -> Self {
        Self {
            vocab_size: 73_744,
            hidden_size: 2048,
            intermediate_size: 7168,
            moe_intermediate_size: 1792,
            num_hidden_layers: 24,
            num_dense_layers: 2,
            num_experts: 32,
            num_experts_per_tok: 4,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            conv_l_cache: 3,
            norm_eps: 1e-5,
            routed_scaling_factor: 1.0,
            full_attention_layers: vec![2, 6, 10, 14, 18, 21],
        }
    }
}

impl Lfm2MoeConfig {
    /// Compact micro-config for rapid in-memory unit tests and CI verification.
    pub fn compact_test() -> Self {
        Self {
            vocab_size: 256,
            hidden_size: 32,
            intermediate_size: 64,
            moe_intermediate_size: 32,
            num_hidden_layers: 4,
            num_dense_layers: 1,
            num_experts: 4,
            num_experts_per_tok: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            conv_l_cache: 3,
            norm_eps: 1e-5,
            routed_scaling_factor: 1.0,
            full_attention_layers: vec![2],
        }
    }
}

/// Hugging Face `config.json` parser for `LiquidAI/LFM2.5-8B-A1B`.
#[derive(Debug, Clone, Deserialize)]
pub struct HfLfm2MoeJsonConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub num_hidden_layers: usize,
    #[serde(default = "default_dense_layers")]
    pub num_dense_layers: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default = "default_conv_cache")]
    pub conv_l_cache: usize,
    pub norm_eps: f64,
    #[serde(default = "default_scaling_factor")]
    pub routed_scaling_factor: f32,
    #[serde(default)]
    pub layer_types: Vec<String>,
    #[serde(default)]
    pub full_attn_idxs: Vec<usize>,
}

fn default_dense_layers() -> usize {
    2
}
fn default_conv_cache() -> usize {
    3
}
fn default_scaling_factor() -> f32 {
    1.0
}

impl From<HfLfm2MoeJsonConfig> for Lfm2MoeConfig {
    fn from(hf: HfLfm2MoeJsonConfig) -> Self {
        let full_attention_layers = if !hf.full_attn_idxs.is_empty() {
            hf.full_attn_idxs
        } else if !hf.layer_types.is_empty() {
            hf.layer_types
                .iter()
                .enumerate()
                .filter_map(|(idx, t)| (t == "full_attention").then_some(idx))
                .collect()
        } else {
            vec![2, 6, 10, 14, 18, 21]
        };

        Self {
            vocab_size: hf.vocab_size,
            hidden_size: hf.hidden_size,
            intermediate_size: hf.intermediate_size,
            moe_intermediate_size: hf.moe_intermediate_size,
            num_hidden_layers: hf.num_hidden_layers,
            num_dense_layers: hf.num_dense_layers,
            num_experts: hf.num_experts,
            num_experts_per_tok: hf.num_experts_per_tok,
            num_attention_heads: hf.num_attention_heads,
            num_key_value_heads: hf.num_key_value_heads,
            conv_l_cache: hf.conv_l_cache,
            norm_eps: hf.norm_eps,
            routed_scaling_factor: hf.routed_scaling_factor,
            full_attention_layers,
        }
    }
}

/// `model.layers.{i}.conv` (`Lfm2MoeShortConv`): Double-gated short-range causal convolution
/// with rolling `[B, D, K-1]` state cache across `80 ms` ticks.
pub struct Lfm2MoeShortConv {
    pub in_proj: Linear,
    pub conv: Conv1d,
    pub out_proj: Linear,
    pub hidden_size: usize,
    pub kernel_size: usize,
}

impl Lfm2MoeShortConv {
    pub fn init(hidden_size: usize, kernel_size: usize, device: &Device) -> Self {
        Self {
            in_proj: LinearConfig::new(hidden_size, hidden_size * 3)
                .with_bias(false)
                .init(device),
            conv: Conv1dConfig::new(hidden_size, hidden_size, kernel_size)
                .with_groups(hidden_size)
                .with_padding(PaddingConfig1d::Explicit(0, 0))
                .with_bias(false)
                .init(device),
            out_proj: LinearConfig::new(hidden_size, hidden_size)
                .with_bias(false)
                .init(device),
            hidden_size,
            kernel_size,
        }
    }

    /// Execute double-gated causal short convolution with rolling `[B, D, K-1]` cache update:
    /// 1. `[B_t, C_t, x_t] = in_proj(h_t)`
    /// 2. `v_t = x_t * SiLU(B_t)`
    /// 3. `u_t = DepthwiseCausalConv1d_K3(concat(conv_cache, v_t))`
    /// 4. `y_t = out_proj(u_t * SiLU(C_t))`
    pub fn forward_with_cache(
        &self,
        x: Tensor<3>,
        conv_cache: Option<Tensor<3>>,
    ) -> (Tensor<3>, Tensor<3>) {
        let [b, _t, d] = x.dims();
        let cache_len = self.kernel_size.saturating_sub(1);
        let device = x.device();

        let proj_bcx = self.in_proj.forward(x);
        let chunks = proj_bcx.chunk(3, 2);
        let (b_gate, c_gate, x_val) = (chunks[0].clone(), chunks[1].clone(), chunks[2].clone());

        let gated_in = (x_val * activation::silu(b_gate)).swap_dims(1, 2);

        let cache = conv_cache.unwrap_or_else(|| Tensor::<3>::zeros([b, d, cache_len], &device));
        let padded = Tensor::cat(vec![cache, gated_in], 2);
        let total_t = padded.dims()[2];

        let next_cache = padded
            .clone()
            .slice([0..b, 0..d, (total_t - cache_len)..total_t]);

        let conv_out = self.conv.forward(padded).swap_dims(1, 2);
        let y = self.out_proj.forward(conv_out * activation::silu(c_gate));
        (y, next_cache)
    }
}

/// `model.layers.{i}.self_attn` (`Lfm2MoeAttention`): Grouped-Query Attention (`32` Q / `8` KV heads)
/// with per-head `QK-RMSNorm`.
pub struct Lfm2MoeAttention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
    pub q_layernorm: RmsNorm,
    pub k_layernorm: RmsNorm,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl Lfm2MoeAttention {
    pub fn init(cfg: &Lfm2MoeConfig, device: &Device) -> Self {
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let kv_dim = cfg.num_key_value_heads * head_dim;
        Self {
            q_proj: LinearConfig::new(cfg.hidden_size, cfg.hidden_size)
                .with_bias(false)
                .init(device),
            k_proj: LinearConfig::new(cfg.hidden_size, kv_dim)
                .with_bias(false)
                .init(device),
            v_proj: LinearConfig::new(cfg.hidden_size, kv_dim)
                .with_bias(false)
                .init(device),
            out_proj: LinearConfig::new(cfg.hidden_size, cfg.hidden_size)
                .with_bias(false)
                .init(device),
            q_layernorm: RmsNormConfig::new(head_dim)
                .with_epsilon(cfg.norm_eps)
                .init(device),
            k_layernorm: RmsNormConfig::new(head_dim)
                .with_epsilon(cfg.norm_eps)
                .init(device),
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim,
        }
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [b, t, d] = x.dims();
        let groups = (self.num_heads / self.num_kv_heads).max(1);
        let q = self
            .q_layernorm
            .forward(
                self.q_proj
                    .forward(x.clone())
                    .reshape([b, t, self.num_heads, self.head_dim]),
            )
            .swap_dims(1, 2);
        let k = self
            .k_layernorm
            .forward(
                self.k_proj
                    .forward(x.clone())
                    .reshape([b, t, self.num_kv_heads, self.head_dim]),
            )
            .repeat_dim(2, groups)
            .reshape([b, t, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = self
            .v_proj
            .forward(x)
            .reshape([b, t, self.num_kv_heads, self.head_dim])
            .repeat_dim(2, groups)
            .reshape([b, t, self.num_heads, self.head_dim])
            .swap_dims(1, 2);

        let scale = (self.head_dim as f32).sqrt();
        let scores = q.matmul(k.swap_dims(2, 3)).div_scalar(scale);
        let attn = activation::softmax(scores, 3);
        let ctx = attn.matmul(v).swap_dims(1, 2).reshape([b, t, d]);
        self.out_proj.forward(ctx)
    }
}

/// SwiGLU MLP (`w1` gate, `w3` up, `w2` down) used in both shallow dense layers (`0..2`)
/// and each of the `32` routed experts in layers `2..24`.
pub struct Lfm2MoeMlp {
    pub w1: Linear,
    pub w2: Linear,
    pub w3: Linear,
}

impl Lfm2MoeMlp {
    pub fn init(hidden_size: usize, intermediate_size: usize, device: &Device) -> Self {
        Self {
            w1: LinearConfig::new(hidden_size, intermediate_size)
                .with_bias(false)
                .init(device),
            w2: LinearConfig::new(intermediate_size, hidden_size)
                .with_bias(false)
                .init(device),
            w3: LinearConfig::new(hidden_size, intermediate_size)
                .with_bias(false)
                .init(device),
        }
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let gate = activation::silu(self.w1.forward(x.clone()));
        let up = self.w3.forward(x);
        self.w2.forward(gate * up)
    }
}

/// `Lfm2MoeSparseMoeBlock`: 32-expert Sparse MoE with normalized sigmoid affinity routing,
/// load-balancing `expert_bias`, and `top_k = 4` active experts per token.
pub struct Lfm2MoeSparseBlock {
    pub gate: Linear,
    pub experts: Vec<Lfm2MoeMlp>,
    pub expert_bias: Param<Tensor<1>>,
    pub top_k: usize,
    pub routed_scaling_factor: f32,
}

impl Lfm2MoeSparseBlock {
    pub fn init(cfg: &Lfm2MoeConfig, device: &Device) -> Self {
        let experts = (0..cfg.num_experts)
            .map(|_| Lfm2MoeMlp::init(cfg.hidden_size, cfg.moe_intermediate_size, device))
            .collect();
        Self {
            gate: LinearConfig::new(cfg.hidden_size, cfg.num_experts)
                .with_bias(false)
                .init(device),
            experts,
            expert_bias: Param::from_tensor(Tensor::zeros([cfg.num_experts], device)),
            top_k: cfg.num_experts_per_tok.min(cfg.num_experts).max(1),
            routed_scaling_factor: cfg.routed_scaling_factor,
        }
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [b, t, _] = x.dims();
        let bias = self.expert_bias.val().reshape([1, 1, self.experts.len()]);
        // Sigmoid affinity routing + bias (LFM2.5-8B-A1B use_expert_bias = true, norm_topk_prob = true)
        let affinities = activation::sigmoid(self.gate.forward(x.clone())) + bias;
        let active = self.top_k.min(self.experts.len());

        let top_slice = affinities.clone().slice([0..b, 0..t, 0..active]);
        let norm_denom = top_slice.sum_dim(2).clamp_min(1e-6);

        let mut acc = Tensor::<3>::zeros(x.dims(), &x.device());
        for (idx, expert) in self.experts.iter().take(active).enumerate() {
            let w = (affinities.clone().slice([0..b, 0..t, idx..(idx + 1)]) / norm_denom.clone())
                * self.routed_scaling_factor;
            acc = acc + expert.forward(x.clone()) * w;
        }
        acc
    }
}

/// Hybrid decoder layer (`Lfm2MoeDecoderLayer`): either `Lfm2MoeShortConv` (18 layers)
/// or `Lfm2MoeAttention` (6 layers), followed by dense SwiGLU (`layers 0..2`) or 32-expert MoE (`layers 2..24`).
pub struct Lfm2MoeDecoderLayer {
    pub operator_norm: RmsNorm,
    pub ffn_norm: RmsNorm,
    pub conv: Option<Lfm2MoeShortConv>,
    pub self_attn: Option<Lfm2MoeAttention>,
    pub dense_mlp: Option<Lfm2MoeMlp>,
    pub moe_mlp: Option<Lfm2MoeSparseBlock>,
}

impl Lfm2MoeDecoderLayer {
    pub fn init(layer_idx: usize, cfg: &Lfm2MoeConfig, device: &Device) -> Self {
        let is_attn = cfg.full_attention_layers.contains(&layer_idx);
        let is_dense = layer_idx < cfg.num_dense_layers;

        Self {
            operator_norm: RmsNormConfig::new(cfg.hidden_size)
                .with_epsilon(cfg.norm_eps)
                .init(device),
            ffn_norm: RmsNormConfig::new(cfg.hidden_size)
                .with_epsilon(cfg.norm_eps)
                .init(device),
            conv: if !is_attn {
                Some(Lfm2MoeShortConv::init(
                    cfg.hidden_size,
                    cfg.conv_l_cache,
                    device,
                ))
            } else {
                None
            },
            self_attn: if is_attn {
                Some(Lfm2MoeAttention::init(cfg, device))
            } else {
                None
            },
            dense_mlp: if is_dense {
                Some(Lfm2MoeMlp::init(
                    cfg.hidden_size,
                    cfg.intermediate_size,
                    device,
                ))
            } else {
                None
            },
            moe_mlp: if !is_dense {
                Some(Lfm2MoeSparseBlock::init(cfg, device))
            } else {
                None
            },
        }
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let h_norm = self.operator_norm.forward(x.clone());
        let op_out = if let Some(conv) = &self.conv {
            conv.forward_with_cache(h_norm, None).0
        } else if let Some(attn) = &self.self_attn {
            attn.forward(h_norm)
        } else {
            h_norm
        };
        let h = x + op_out;
        let ffn_in = self.ffn_norm.forward(h.clone());
        let ffn_out = if let Some(dense) = &self.dense_mlp {
            dense.forward(ffn_in)
        } else if let Some(moe) = &self.moe_mlp {
            moe.forward(ffn_in)
        } else {
            ffn_in
        };
        h + ffn_out
    }
}

/// Top-level `Lfm2MoeModel` (`LiquidAI/LFM2.5-8B-A1B`) equipped with:
/// - `lm_head`: text + $p=1$ semantic audio token predictor
/// - `vad_head`: 8-class multi-horizon Semantic VAD head (`80 ms` clock)
/// - `kokoro_prosody_proj`: projects $\mathbf{h}_t \in \mathbb{R}^{2048}$ into `Kokoro-82M` `512`-d acoustic latents
pub struct Lfm2MoeModel {
    pub embed_tokens: Embedding,
    pub layers: Vec<Lfm2MoeDecoderLayer>,
    pub embedding_norm: RmsNorm,
    pub lm_head: Linear,
    pub vad_head: Linear,
    pub kokoro_prosody_proj: Linear,
}

impl Lfm2MoeModel {
    pub fn init(cfg: &Lfm2MoeConfig, device: &Device) -> Self {
        let layers = (0..cfg.num_hidden_layers)
            .map(|idx| Lfm2MoeDecoderLayer::init(idx, cfg, device))
            .collect();

        Self {
            embed_tokens: EmbeddingConfig::new(cfg.vocab_size, cfg.hidden_size).init(device),
            layers,
            embedding_norm: RmsNormConfig::new(cfg.hidden_size)
                .with_epsilon(cfg.norm_eps)
                .init(device),
            lm_head: LinearConfig::new(cfg.hidden_size, cfg.vocab_size)
                .with_bias(false)
                .init(device),
            vad_head: LinearConfig::new(cfg.hidden_size, 8)
                .with_bias(true)
                .init(device),
            kokoro_prosody_proj: LinearConfig::new(cfg.hidden_size, 512)
                .with_bias(true)
                .init(device),
        }
    }

    pub fn forward_hidden(&self, token_ids: Tensor<2, Int>) -> Tensor<3> {
        let mut h = self.embed_tokens.forward(token_ids);
        for layer in &self.layers {
            h = layer.forward(h);
        }
        self.embedding_norm.forward(h)
    }
}

/// 8-Codebook RVQ Audio Depthformer (`q = 8` codebooks per `80 ms` frame, `0..2047` + stop `2048`).
pub struct DepthformerQ8Head {
    pub codebook_embeds: Vec<Embedding>,
    pub rms_norms: Vec<RmsNorm>,
    pub gate_projs: Vec<Linear>,
    pub up_projs: Vec<Linear>,
    pub codebook_vocab: usize,
}

impl DepthformerQ8Head {
    pub fn init(hidden_dim: usize, codebook_vocab: usize, device: &Device) -> Self {
        let mut codebook_embeds = Vec::with_capacity(8);
        let mut rms_norms = Vec::with_capacity(8);
        let mut gate_projs = Vec::with_capacity(8);
        let mut up_projs = Vec::with_capacity(8);

        for _ in 0..8 {
            codebook_embeds.push(EmbeddingConfig::new(codebook_vocab, hidden_dim).init(device));
            rms_norms.push(RmsNormConfig::new(hidden_dim).with_epsilon(1e-5).init(device));
            gate_projs.push(
                LinearConfig::new(hidden_dim, codebook_vocab)
                    .with_bias(false)
                    .init(device),
            );
            up_projs.push(
                LinearConfig::new(hidden_dim, codebook_vocab)
                    .with_bias(false)
                    .init(device),
            );
        }

        Self {
            codebook_embeds,
            rms_norms,
            gate_projs,
            up_projs,
            codebook_vocab,
        }
    }

    /// Autoregressively predicts all `q = 8` RVQ acoustic codebooks (`0..2047`, stop `2048`)
    /// from the 8B trunk hidden state $\mathbf{h}_t$.
    pub fn sample_8_codebooks(&self, trunk_h_t: Tensor<2>) -> [u16; 8] {
        let [b, d] = trunk_h_t.dims();
        let device = trunk_h_t.device();
        let mut state_h = trunk_h_t;
        let mut prev_tok = Tensor::<2, Int>::zeros([b, 1], &device);
        let mut codes = [0u16; 8];

        for k in 0..8 {
            if k > 0 {
                let emb = self.codebook_embeds[k]
                    .forward(prev_tok.clone())
                    .reshape([b, d]);
                state_h = state_h + emb;
            }
            let norm_h = self.rms_norms[k]
                .forward(state_h.clone().reshape([b, 1, d]))
                .reshape([b, d]);
            let gate = activation::silu(self.gate_projs[k].forward(norm_h.clone()));
            let up = self.up_projs[k].forward(norm_h);
            let logits = gate * up;
            let next_tok = logits.argmax(1);
            let tok_val = next_tok
                .clone()
                .into_data()
                .try_to_vec::<i64>()
                .ok()
                .and_then(|v| v.first().copied())
                .unwrap_or(0) as u16;
            codes[k] = tok_val % (self.codebook_vocab as u16);
            prev_tok = next_tok;
        }
        codes
    }
}
