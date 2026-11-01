use std::collections::HashMap;

use anyhow::{ bail, Context, Result };
use serde::{ Deserialize, Serialize };

use crate::kokoro_modules::{ KOKORO_82M_KEY_REMAP_RULES, LFM2_MOE_KEY_REMAP_RULES };

/// Official Hugging Face repository IDs fused into `Aurora-8B-Ultra-RL`.
pub const HF_SOURCE_REPOS: &[(&str, &str)] = &[
    ("backbone_lfm2_moe", "LiquidAI/LFM2.5-8B-A1B"),
    ("teacher_mimo_rl", "XiaomiMiMo/MiMo-V2.6-Pro-RL"),
    ("asr_vad_infinite", "Edge0/Audio8-ASR-Infinite"),
    ("tts_arktts_head", "Edge0/Audio8-TTS-Realistic-0.6b"),
    ("tts_kokoro_82m", "hexgrad/Kokoro-82M"),
];

/// Tensor descriptor inside a `.safetensors` JSON header.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SafetensorTensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [usize; 2],
}

/// Parsed `.safetensors` header with remapped Burn module parameter keys.
#[derive(Debug, Clone)]
pub struct ParsedSafetensorsHeader {
    pub raw_header_bytes: usize,
    pub tensors: HashMap<String, SafetensorTensorInfo>,
}

/// Zero-Python `.safetensors` binary header parser (`8-byte u64 LE length` + JSON tensor table).
pub fn parse_safetensors_header(bytes: &[u8]) -> Result<ParsedSafetensorsHeader> {
    if bytes.len() < 8 {
        bail!("safetensors buffer too short (< 8 bytes)");
    }
    let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    if bytes.len() < 8 + header_len {
        bail!(
            "safetensors buffer truncated: header_len={} but total_len={}",
            header_len,
            bytes.len()
        );
    }

    let json_slice = &bytes[8..8 + header_len];
    let raw_map: HashMap<String, serde_json::Value> = serde_json::from_slice(json_slice)
        .context("invalid JSON in safetensors header")?;

    let mut tensors = HashMap::new();
    for (k, v) in raw_map {
        if k == "__metadata__" {
            continue;
        }
        let info: SafetensorTensorInfo = serde_json::from_value(v)
            .with_context(|| format!("invalid tensor descriptor for key {k}"))?;
        tensors.insert(k, info);
    }

    Ok(ParsedSafetensorsHeader {
        raw_header_bytes: header_len,
        tensors,
    })
}

/// Remaps a PyTorch `state_dict` tensor key from `LiquidAI/LFM2.5-8B-A1B` or `hexgrad/Kokoro-82M`
/// into its canonical Burn `Aurora-8B-Ultra-RL` module path.
pub fn remap_pytorch_key_to_burn(key: &str) -> String {
    if let Some(rest) = key.strip_prefix("model.layers.") {
        if let Some((layer_idx, tail)) = rest.split_once('.') {
            if tail == "feed_forward.gate.weight" {
                return format!("layers.{layer_idx}.moe_mlp.gate.weight");
            }
            if tail == "feed_forward.expert_bias" {
                return format!("layers.{layer_idx}.moe_mlp.expert_bias");
            }
            if let Some(exp_rest) = tail.strip_prefix("feed_forward.experts.") {
                return format!("layers.{layer_idx}.moe_mlp.experts.{exp_rest}");
            }
            if let Some(dense_rest) = tail.strip_prefix("feed_forward.") {
                return format!("layers.{layer_idx}.dense_mlp.{dense_rest}");
            }
            return format!("layers.{layer_idx}.{tail}");
        }
    }

    if let Some(rest) = key.strip_prefix("model.") {
        return rest.to_string();
    }

    if key == "decoder.generator.m_source.l_linear.weight" {
        return "decoder.generator.m_source_l_linear.weight".to_string();
    }
    if key == "decoder.generator.m_source.l_linear.bias" {
        return "decoder.generator.m_source_l_linear.bias".to_string();
    }
    if let Some(bert_rest) = key.strip_prefix("bert.encoder.layer.") {
        return format!("plbert.layers.{bert_rest}");
    }

    let _ = (LFM2_MOE_KEY_REMAP_RULES.len(), KOKORO_82M_KEY_REMAP_RULES.len());
    key.to_string()
}

/// Builds a synthetic in-memory `.safetensors` byte buffer for testing zero-Python weight ingestion.
pub fn build_test_safetensors_blob(tensors: &[(&str, &[usize], &[f32])]) -> Vec<u8> {
    let mut header_map = serde_json::Map::new();
    let mut data_buf = Vec::new();

    for &(name, shape, values) in tensors {
        let start = data_buf.len();
        for &val in values {
            data_buf.extend_from_slice(&val.to_le_bytes());
        }
        let end = data_buf.len();
        header_map.insert(
            name.to_string(),
            serde_json::json!({
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [start, end],
            }),
        );
    }

    let header_bytes = serde_json::to_vec(&header_map).unwrap();
    let header_len = header_bytes.len() as u64;
    let mut out = Vec::with_capacity(8 + header_bytes.len() + data_buf.len());
    out.extend_from_slice(&header_len.to_le_bytes());
    out.extend_from_slice(&header_bytes);
    out.extend_from_slice(&data_buf);
    out
}
