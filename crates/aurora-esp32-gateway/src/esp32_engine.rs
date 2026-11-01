use std::{net::SocketAddr, sync::Arc};

use aurora_core::DecodedEsp32Frame;
use aurora_kokoro_tts::KokoroIstftGenerator;
use burn::{
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig},
    tensor::{activation, Device, Int, Tensor},
};
use dashmap::DashMap;

/// Lifts the ESP32-S3's 2-codebook RVQ indices (`rvq_q0[10]`, `rvq_q1[8]`, codes `0..2047`)
/// into the `LFM2.5-8B-A1B` hidden representation and 8-class Semantic VAD logits.
pub struct Esp32QuantalAdapter {
    pub q0_embed: Embedding,
    pub q1_embed: Embedding,
    pub semantic_vad_head: Linear,
}

impl Esp32QuantalAdapter {
    pub fn init(hidden_dim: usize, device: &Device) -> Self {
        Self {
            q0_embed: EmbeddingConfig::new(2048, hidden_dim).init(device),
            q1_embed: EmbeddingConfig::new(2048, hidden_dim).init(device),
            semantic_vad_head: LinearConfig::new(hidden_dim, 8).init(device),
        }
    }

    pub fn lift_packet_to_lfm_hidden(
        &self,
        frame: &DecodedEsp32Frame,
        device: &Device,
    ) -> (Tensor<3>, Tensor<2>) {
        let q0_ids: Vec<i64> = frame.rvq_q0.iter().map(|&c| (c as i64).clamp(0, 2047)).collect();
        let q1_ids: Vec<i64> = frame.rvq_q1.iter().map(|&c| (c as i64).clamp(0, 2047)).collect();

        let q0_tensor = Tensor::<1, Int>::from_ints(q0_ids.as_slice(), device).reshape([1, 10]);
        let q1_tensor = Tensor::<1, Int>::from_ints(q1_ids.as_slice(), device).reshape([1, 8]);

        let h_q0 = self.q0_embed.forward(q0_tensor).mean_dim(1); // [1, 1, D]
        let h_q1 = self.q1_embed.forward(q1_tensor).mean_dim(1); // [1, 1, D]
        let d = h_q0.dims()[2];

        let fused = activation::silu(h_q0 + h_q1); // [1, 1, D]
        let vad_logits = self
            .semantic_vad_head
            .forward(fused.clone().reshape([1, d]));
        (fused, vad_logits)
    }
}

/// Per-ESP32 node rolling state tracked across `80 ms` WebSocket frames.
pub struct Esp32DeviceSession {
    pub peer: SocketAddr,
    pub last_seq: u16,
    pub has_uncommitted_speech: bool,
    pub silent_ticks: usize,
    pub frames_in_turn: usize,
}

impl Esp32DeviceSession {
    pub fn new(peer: SocketAddr) -> Self {
        Self {
            peer,
            last_seq: 0,
            has_uncommitted_speech: false,
            silent_ticks: 0,
            frames_in_turn: 0,
        }
    }
}

/// Decision returned by `Esp32GatewayEngine::step_80ms_packet`.
#[derive(Debug, Clone)]
pub enum Esp32StepOutcome {
    /// User interrupted while TTS is playing -> cancel active egress immediately.
    BargeInInterrupt,
    /// Frame accumulated in rolling state (`< 5` silent ticks or active speech).
    Listening { seq: u16 },
    /// End-of-turn committed -> emit 24 kHz Kokoro waveform + optional JSON GPIO command.
    TurnCommitted {
        seq: u16,
        transcript: String,
        gpio_json: Option<String>,
        kokoro_pcm_24k: Arc<Vec<f32>>,
    },
}

/// Standalone or embedded engine managing connected ESP32-S3 nodes.
pub struct Esp32GatewayEngine {
    pub device: Device,
    pub adapter: Esp32QuantalAdapter,
    pub kokoro_gen: KokoroIstftGenerator,
    pub sessions: DashMap<SocketAddr, Esp32DeviceSession>,
}

impl Esp32GatewayEngine {
    pub fn new(device: Device) -> Arc<Self> {
        let adapter = Esp32QuantalAdapter::init(64, &device);
        let kokoro_gen = KokoroIstftGenerator::init_scaled(32, 8, &device);
        Arc::new(Self {
            device,
            adapter,
            kokoro_gen,
            sessions: DashMap::new(),
        })
    }

    pub fn step_80ms_packet(
        &self,
        peer: SocketAddr,
        frame: DecodedEsp32Frame,
        tts_currently_active: bool,
    ) -> Esp32StepOutcome {
        let (_fused_h, _vad_logits) = self.adapter.lift_packet_to_lfm_hidden(&frame, &self.device);
        let mut session = self
            .sessions
            .entry(peer)
            .or_insert_with(|| Esp32DeviceSession::new(peer));

        session.last_seq = frame.seq;
        let is_speech = frame.vad_prob > 0.50;

        if is_speech {
            session.has_uncommitted_speech = true;
            session.silent_ticks = 0;
            session.frames_in_turn += 1;
            if tts_currently_active && frame.vad_prob > 0.65 {
                return Esp32StepOutcome::BargeInInterrupt;
            }
            return Esp32StepOutcome::Listening { seq: frame.seq };
        }

        if !session.has_uncommitted_speech {
            return Esp32StepOutcome::Listening { seq: frame.seq };
        }

        session.silent_ticks += 1;
        // Hold turn open during < 400 ms (< 5 ticks * 80 ms) mid-sentence thinking pauses
        if session.silent_ticks < 5 {
            return Esp32StepOutcome::Listening { seq: frame.seq };
        }

        // Commit turn -> synthesize 24 kHz response via KokoroIstftGenerator
        session.has_uncommitted_speech = false;
        session.silent_ticks = 0;
        let spoken_frames = session.frames_in_turn;
        session.frames_in_turn = 0;

        let x = Tensor::<3>::full([1, 64, 4], 0.15, &self.device);
        let s = Tensor::<2>::full([1, 32], 0.10, &self.device);
        let f0 = Tensor::<3>::full([1, 1, 8], 215.0, &self.device);
        let wav_24k = self.kokoro_gen.forward(x, s, f0);
        let pcm_24k = wav_24k
            .into_data()
            .try_to_vec::<f32>()
            .unwrap_or_else(|_| vec![0.0f32; 1200]);

        Esp32StepOutcome::TurnCommitted {
            seq: frame.seq,
            transcript: format!("[ESP32 {peer} clause: {spoken_frames} frames]"),
            gpio_json: Some(r#"{"cmd":"gpio","pin":4,"val":1}"#.to_string()),
            kokoro_pcm_24k: Arc::new(pcm_24k),
        }
    }
}
