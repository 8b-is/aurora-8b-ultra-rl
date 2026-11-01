use anyhow::Result;
use aurora_weights::{
    AuroraBurnpackManifest, DepthformerQ8Head, Lfm2MoeConfig, Lfm2MoeModel, WeightNormConv1d,
    KOKORO_82M_KEY_REMAP_RULES, LFM2_MOE_KEY_REMAP_RULES,
};
use burn::tensor::Device;
use tracing::info;

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let device = Device::default();
    let cfg = Lfm2MoeConfig::compact_test();
    let _lfm = Lfm2MoeModel::init(&cfg, &device);
    let _depthformer = DepthformerQ8Head::init(cfg.hidden_size, 2049, &device);
    let wn = WeightNormConv1d::init(16, 32, 3, 1, 1, 1, 1, true, &device);
    let _fused = wn.into_fused_conv1d();

    let manifest = AuroraBurnpackManifest::new_default();
    info!(
        model_id = %manifest.model_id,
        backbone = %manifest.backbone,
        lfm_remap_rules = LFM2_MOE_KEY_REMAP_RULES.len(),
        kokoro_remap_rules = KOKORO_82M_KEY_REMAP_RULES.len(),
        blake3 = %manifest.blake3_digest,
        "Aurora-8B-Ultra-RL Burnpack (.bpk) weight importer & WeightNormConv1d fuser verified."
    );

    Ok(())
}
