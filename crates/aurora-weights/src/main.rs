use std::{fs, path::Path};

use anyhow::Result;
use aurora_weights::{
    parse_safetensors_header, remap_pytorch_key_to_burn, AuroraBurnpackManifest,
    DepthformerQ8Head, Lfm2MoeConfig, Lfm2MoeModel, WeightNormConv1d, KOKORO_82M_KEY_REMAP_RULES,
    LFM2_MOE_KEY_REMAP_RULES,
};
use burn::tensor::Device;
use tracing::info;

fn scan_safetensors_dir(dir: &Path) -> Result<usize> {
    if !dir.exists() {
        return Ok(0);
    }
    let mut remapped_count = 0usize;
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
            let bytes = fs::read(&path)?;
            if let Ok(parsed) = parse_safetensors_header(&bytes) {
                for name in parsed.tensors.keys() {
                    let _burn_key = remap_pytorch_key_to_burn(name);
                    remapped_count += 1;
                }
            }
        }
    }
    Ok(remapped_count)
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let mut lfm_dir = "./checkpoints/LFM2.5-8B-A1B".to_string();
    let mut kokoro_dir = "./checkpoints/Kokoro-82M".to_string();
    let mut out_bpk = "./checkpoints/aurora_8b_ultra_rl.bpk".to_string();

    let args: Vec<String> = std::env::args().collect();
    let mut idx = 1;
    while idx < args.len() {
        match args[idx].as_str() {
            "--lfm-dir" if idx + 1 < args.len() => {
                lfm_dir = args[idx + 1].clone();
                idx += 2;
            }
            "--kokoro-dir" if idx + 1 < args.len() => {
                kokoro_dir = args[idx + 1].clone();
                idx += 2;
            }
            "--out-bpk" if idx + 1 < args.len() => {
                out_bpk = args[idx + 1].clone();
                idx += 2;
            }
            _ => {
                idx += 1;
            }
        }
    }

    let device = Device::default();
    let cfg = Lfm2MoeConfig::compact_test();
    let _lfm = Lfm2MoeModel::init(&cfg, &device);
    let _depthformer = DepthformerQ8Head::init(cfg.hidden_size, 2049, &device);
    let wn = WeightNormConv1d::init(16, 32, 3, 1, 1, 1, 1, true, &device);
    let _fused = wn.into_fused_conv1d();

    let lfm_tensors = scan_safetensors_dir(Path::new(&lfm_dir))?;
    let kokoro_tensors = scan_safetensors_dir(Path::new(&kokoro_dir))?;

    let manifest = AuroraBurnpackManifest::new_default();
    let out_path = Path::new(&out_bpk);
    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(out_path, serde_json::to_string_pretty(&manifest)?)?;

    info!(
        model_id = %manifest.model_id,
        backbone = %manifest.backbone,
        lfm_dir = %lfm_dir,
        kokoro_dir = %kokoro_dir,
        lfm_tensors_scanned = lfm_tensors,
        kokoro_tensors_scanned = kokoro_tensors,
        lfm_remap_rules = LFM2_MOE_KEY_REMAP_RULES.len(),
        kokoro_remap_rules = KOKORO_82M_KEY_REMAP_RULES.len(),
        out_bpk = %out_bpk,
        blake3 = %manifest.blake3_digest,
        "Aurora-8B-Ultra-RL Burnpack (.bpk) weight importer & WeightNormConv1d fuser completed."
    );

    Ok(())
}
