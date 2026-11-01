use burn::{
    module::Param,
    nn::conv::{Conv1d, Conv1dConfig},
    nn::PaddingConfig1d,
    tensor::{module::conv1d, ops::ConvOptions, Device, Tensor},
};

/// Regex key-remapping rules translating `LiquidAI/LFM2.5-8B-A1B` PyTorch `safetensors` keys
/// into the `Lfm2MoeModel` Burn module tree.
pub const LFM2_MOE_KEY_REMAP_RULES: &[(&str, &str)] = &[
    (r"^model\.layers\.(\d+)\.feed_forward\.gate\.weight$", "layers.$1.moe_mlp.gate.weight"),
    (r"^model\.layers\.(\d+)\.feed_forward\.expert_bias$", "layers.$1.moe_mlp.expert_bias"),
    (
        r"^model\.layers\.(\d+)\.feed_forward\.experts\.(\d+)\.(w1|w2|w3)\.weight$",
        "layers.$1.moe_mlp.experts.$2.$3.weight",
    ),
    (
        r"^model\.layers\.(\d+)\.feed_forward\.(w1|w2|w3)\.weight$",
        "layers.$1.dense_mlp.$2.weight",
    ),
    (r"^model\.layers\.(\d+)\.conv\.conv\.weight$", "layers.$1.conv.conv.weight"),
    (
        r"^model\.layers\.(\d+)\.conv\.(in_proj|out_proj)\.weight$",
        "layers.$1.conv.$2.weight",
    ),
    (r"^model\.(.*)$", "$1"),
];

/// Regex key-remapping rules translating `hexgrad/Kokoro-82M` (`kokoro-v1_0.safetensors`)
/// PyTorch `state_dict` keys into the Burn `KokoroIstftGenerator` hierarchy.
pub const KOKORO_82M_KEY_REMAP_RULES: &[(&str, &str)] = &[
    (r"^decoder\.encode\.(.*)$", "decoder.encode.$1"),
    (r"^decoder\.decode\.(\d+)\.(.*)$", "decoder.decode.$1.$2"),
    (r"^decoder\.generator\.ups\.(\d+)\.(.*)$", "decoder.generator.ups.$1.$2"),
    (r"^decoder\.generator\.resblocks\.(\d+)\.(.*)$", "decoder.generator.resblocks.$1.$2"),
    (r"^decoder\.generator\.noise_convs\.(\d+)\.(.*)$", "decoder.generator.noise_convs.$1.$2"),
    (r"^decoder\.generator\.noise_res\.(\d+)\.(.*)$", "decoder.generator.noise_res.$1.$2"),
    (
        r"^decoder\.generator\.m_source\.l_linear\.(weight|bias)$",
        "decoder.generator.m_source_l_linear.$1",
    ),
    (r"^predictor\.text_encoder\.(.*)$", "predictor.text_encoder.$1"),
    (r"^bert\.encoder\.layer\.(\d+)\.(.*)$", "plbert.layers.$1.$2"),
];

/// `Kokoro-82M` `torch.nn.utils.weight_norm` 1D convolution (`weight_g`: `[out, 1, 1]`,
/// `weight_v`: `[out, in/groups, kernel]`) with one-time `into_fused_conv1d()` export.
pub struct WeightNormConv1d {
    pub weight_g: Param<Tensor<3>>,
    pub weight_v: Param<Tensor<3>>,
    pub bias: Option<Param<Tensor<1>>>,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub groups: usize,
}

impl WeightNormConv1d {
    pub fn init(
        in_c: usize,
        out_c: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
        has_bias: bool,
        device: &Device,
    ) -> Self {
        let base_conv = Conv1dConfig::new(in_c, out_c, kernel)
            .with_stride(stride)
            .with_padding(PaddingConfig1d::Explicit(padding, padding))
            .with_dilation(dilation)
            .with_groups(groups)
            .with_bias(has_bias)
            .init(device);
        Self {
            weight_g: Param::from_tensor(Tensor::ones([out_c, 1, 1], device)),
            weight_v: base_conv.weight,
            bias: base_conv.bias,
            stride,
            padding,
            dilation,
            groups,
        }
    }

    /// Computes $W = g \cdot \frac{v}{\|v\|_2 + 10^{-12}}$ across `[in/groups, kernel]`.
    pub fn compute_normalized_weight(&self) -> Tensor<3> {
        let v = self.weight_v.val();
        let g = self.weight_g.val();
        let norm_v = v
            .clone()
            .powf_scalar(2.0)
            .sum_dim(1)
            .sum_dim(2)
            .sqrt()
            .clamp_min(1e-12);
        v * (g / norm_v)
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let w = self.compute_normalized_weight();
        conv1d(
            x,
            w,
            self.bias.as_ref().map(|b| b.val()),
            ConvOptions::new([self.stride], [self.padding], [self.dilation], self.groups),
        )
    }

    /// Fuses `weight_g` and `weight_v` into a single standard Burn `Conv1d` prior to `.bpk` export.
    pub fn into_fused_conv1d(self) -> Conv1d {
        let fused_weight = self.compute_normalized_weight();
        let kernel_size = fused_weight.dims()[2];
        Conv1d {
            weight: Param::from_tensor(fused_weight),
            bias: self.bias,
            stride: self.stride,
            kernel_size,
            dilation: self.dilation,
            groups: self.groups,
            padding: PaddingConfig1d::Explicit(self.padding, self.padding),
        }
    }
}
