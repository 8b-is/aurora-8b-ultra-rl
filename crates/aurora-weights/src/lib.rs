pub mod hf_loader;
pub mod kokoro_modules;
pub mod lfm2_moe;

pub use hf_loader::{
    build_test_safetensors_blob, parse_safetensors_header, remap_pytorch_key_to_burn,
    ParsedSafetensorsHeader, SafetensorTensorInfo, HF_SOURCE_REPOS,
};
pub use kokoro_modules::{WeightNormConv1d, KOKORO_82M_KEY_REMAP_RULES, LFM2_MOE_KEY_REMAP_RULES};
pub use lfm2_moe::{
    DepthformerQ8Head, HfLfm2MoeJsonConfig, Lfm2MoeAttention, Lfm2MoeConfig, Lfm2MoeDecoderLayer,
    Lfm2MoeMlp, Lfm2MoeModel, Lfm2MoeShortConv, Lfm2MoeSparseBlock,
};

use serde::{Deserialize, Serialize};

/// Metadata header written into exported `.bpk` Burnpack archives for zero-copy `mmap` loading.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuroraBurnpackManifest {
    pub format_version: u32,
    pub model_id: String,
    pub backbone: String,
    pub acoustic_generator: String,
    pub total_params: u64,
    pub active_params_per_token: u64,
    pub quantal_p_semantic: u8,
    pub quantal_q_rvq_codebooks: u8,
    pub weight_norm_fused: bool,
    pub blake3_digest: String,
}

impl AuroraBurnpackManifest {
    pub fn new_default() -> Self {
        let payload = b"Aurora-8B-Ultra-RL:LFM2.5-8B-A1B+Kokoro-82M+Audio8-DepthformerQ8";
        let digest = blake3::hash(payload).to_hex().to_string();
        Self {
            format_version: 1,
            model_id: "8b-is/Aurora-8B-Ultra-RL".to_string(),
            backbone: "LiquidAI/LFM2.5-8B-A1B".to_string(),
            acoustic_generator: "hexgrad/Kokoro-82M + Edge0/Audio8-TTS-Realistic-0.6b".to_string(),
            total_params: 8_380_000_000,
            active_params_per_token: 1_520_000_000,
            quantal_p_semantic: 1,
            quantal_q_rvq_codebooks: 8,
            weight_norm_fused: true,
            blake3_digest: digest,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Int, Tensor};

    #[test]
    fn test_lfm2_moe_hybrid_forward_and_depthformer_q8() {
        let device = Device::default();
        let cfg = Lfm2MoeConfig::compact_test();
        let model = Lfm2MoeModel::init(&cfg, &device);
        let depthformer = DepthformerQ8Head::init(cfg.hidden_size, 2049, &device);

        let tokens = Tensor::<1, Int>::from_ints([1, 5, 12, 42], &device).reshape([1, 4]);
        let h = model.forward_hidden(tokens);
        assert_eq!(h.dims(), [1, 4, cfg.hidden_size]);

        let h_last = h
            .slice([0..1, 3..4, 0..cfg.hidden_size])
            .reshape([1, cfg.hidden_size]);
        let codes = depthformer.sample_8_codebooks(h_last);
        assert_eq!(codes.len(), 8);
        for c in codes {
            assert!(c < 2049);
        }
    }

    #[test]
    fn test_weight_norm_conv1d_fusion_matches_dynamic_forward() {
        let device = Device::default();
        let wn = WeightNormConv1d::init(8, 16, 3, 1, 1, 1, 1, true, &device);
        let x = Tensor::<3>::ones([1, 8, 10], &device);
        let y_dynamic = wn.forward(x.clone());
        let fused = wn.into_fused_conv1d();
        let y_fused = fused.forward(x);
        let diff: f32 = (y_dynamic - y_fused)
            .abs()
            .max()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap()[0];
        assert!(diff < 1e-5, "fused Conv1d must match WeightNormConv1d within 1e-5");
    }

    #[test]
    fn test_safetensors_header_parser_and_pytorch_key_remapper() {
        let blob = build_test_safetensors_blob(&[
            ("model.layers.4.feed_forward.experts.7.w1.weight", &[4, 4], &[0.1f32; 16]),
            ("decoder.generator.m_source.l_linear.weight", &[1, 9], &[0.2f32; 9]),
        ]);
        let parsed = parse_safetensors_header(&blob).expect("valid safetensors header");
        assert_eq!(parsed.tensors.len(), 2);

        let mapped_moe =
            remap_pytorch_key_to_burn("model.layers.4.feed_forward.experts.7.w1.weight");
        assert_eq!(mapped_moe, "layers.4.moe_mlp.experts.7.w1.weight");

        let mapped_kokoro =
            remap_pytorch_key_to_burn("decoder.generator.m_source.l_linear.weight");
        assert_eq!(mapped_kokoro, "decoder.generator.m_source_l_linear.weight");
    }
}
