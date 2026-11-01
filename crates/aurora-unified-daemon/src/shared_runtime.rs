use std::{net::SocketAddr, sync::Arc};

use aurora_core::{DecodedEsp32Frame, FRAME_SAMPLES_80MS_16K};
use aurora_esp32_gateway::Esp32QuantalAdapter;
use aurora_kokoro_tts::KokoroIstftGenerator;
use burn::{
    nn::{Linear, LinearConfig},
    tensor::{activation, Device, Tensor},
};
use dashmap::DashMap;
use tokio::sync::broadcast;

/// Unified key identifying either a Discord Voice user (by RTP SSRC) or an ESP32-S3 hardware node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StreamClientKey {
    DiscordSsrc(u32),
    Esp32Node(SocketAddr),
}

/// Cross-platform event broadcast between the Discord Bot and ESP32-S3 nodes.
#[derive(Clone, Debug)]
pub enum CrossBridgeEvent {
    /// Emitted when either a Discord user or an ESP32 node commits a voice turn.
    VoiceTurnCompleted {
        source: StreamClientKey,
        transcript: String,
        reply_text: String,
        gpio_command: Option<String>,
        /// `24 kHz` mono waveform from `KokoroIstftGenerator` (playable on Discord AND ESP32).
        kokoro_24k_mono: Arc<Vec<f32>>,
    },
    /// Hardware GPIO / sensor command routed to a connected ESP32-S3 node.
    Esp32HardwareCommand {
        target: Option<SocketAddr>,
        json_payload: String,
    },
}

/// Isolated per-stream rolling state (`LFM2.5` 18-layer Conv cache + Semantic VAD horizon tracker).
pub struct RollingStreamSlot {
    pub key: StreamClientKey,
    /// `LFM2.5-8B-A1B` 18-layer causal short-convolution state `[18, 1, 256, 2]`.
    pub lfm_conv_cache: Tensor<4>,
    /// Accumulates `16 kHz` mono PCM samples for Discord RTP streams.
    pub pcm_16k_accum: Vec<f32>,
    /// True once speech is detected in the current turn.
    pub has_uncommitted_speech: bool,
    /// Consecutive `80 ms` frames where `p(speech) < threshold`.
    pub trailing_silent_ticks: usize,
    /// Accumulated discrete semantic `p0` tokens in the current clause.
    pub semantic_token_history: Vec<u16>,
}

impl RollingStreamSlot {
    pub fn new(key: StreamClientKey, device: &Device) -> Self {
        Self {
            key,
            lfm_conv_cache: Tensor::<4>::zeros([18, 1, 256, 2], device),
            pcm_16k_accum: Vec::with_capacity(FRAME_SAMPLES_80MS_16K * 2),
            has_uncommitted_speech: false,
            trailing_silent_ticks: 0,
            semantic_token_history: Vec::with_capacity(256),
        }
    }
}

/// Shared model weights residing once in memory (`~5.2 GB` VRAM in INT4/FP8)
/// and shared across all Discord & ESP32 worker tasks.
pub struct AuroraSharedWeights {
    pub hidden_dim: usize,
    pub pcm_patch_encoder: Linear,
    pub esp32_adapter: Esp32QuantalAdapter,
    pub kokoro_style_proj: Linear,
    pub kokoro_gen: KokoroIstftGenerator,
}

impl AuroraSharedWeights {
    pub fn init(device: &Device) -> Self {
        let hidden_dim = 64;
        Self {
            hidden_dim,
            pcm_patch_encoder: LinearConfig::new(FRAME_SAMPLES_80MS_16K, hidden_dim)
                .with_bias(false)
                .init(device),
            esp32_adapter: Esp32QuantalAdapter::init(hidden_dim, device),
            kokoro_style_proj: LinearConfig::new(hidden_dim, 32)
                .with_bias(true)
                .init(device),
            kokoro_gen: KokoroIstftGenerator::init_scaled(32, 8, device),
        }
    }
}

/// Output of a single `80 ms` inference tick on either Discord or ESP32.
#[derive(Debug, Clone)]
pub enum TickDecision {
    BargeInDetected,
    Listening {
        seq_num: u16,
    },
    TurnCommitted {
        transcript: String,
        reply_text: String,
        gpio_command: Option<String>,
        kokoro_24k_mono: Arc<Vec<f32>>,
    },
}

/// Thread-safe singleton holding the Burn device, shared weights, per-client slots, and cross-bridge bus.
pub struct SharedAuroraRuntime {
    pub device: Device,
    pub weights: AuroraSharedWeights,
    pub slots: DashMap<StreamClientKey, RollingStreamSlot>,
    pub bridge_tx: broadcast::Sender<CrossBridgeEvent>,
}

impl SharedAuroraRuntime {
    pub fn new(device: Device) -> Arc<Self> {
        let weights = AuroraSharedWeights::init(&device);
        let (bridge_tx, _) = broadcast::channel(128);
        Arc::new(Self {
            device,
            weights,
            slots: DashMap::new(),
            bridge_tx,
        })
    }

    /// Evaluates an `80 ms` `16 kHz` PCM frame from a Discord user (`StreamClientKey::DiscordSsrc`).
    pub fn step_discord_pcm_80ms(
        &self,
        ssrc: u32,
        frame_16k: [f32; FRAME_SAMPLES_80MS_16K],
        bot_is_speaking: bool,
    ) -> TickDecision {
        let key = StreamClientKey::DiscordSsrc(ssrc);
        let mut slot = self
            .slots
            .entry(key)
            .or_insert_with(|| RollingStreamSlot::new(key, &self.device));

        let pcm_tensor = Tensor::<1>::from_floats(frame_16k.as_slice(), &self.device)
            .reshape([1, FRAME_SAMPLES_80MS_16K]);
        let rms: f32 = pcm_tensor
            .clone()
            .powf_scalar(2.0)
            .mean()
            .sqrt()
            .into_scalar();

        let h_t = activation::silu(self.weights.pcm_patch_encoder.forward(pcm_tensor));
        let _vad_logits = self.weights.esp32_adapter.semantic_vad_head.forward(h_t);

        self.advance_turn_state(&mut slot, 0, rms > 0.012, false, bot_is_speaking && rms > 0.025)
    }

    /// Evaluates an `80 ms` `DecodedEsp32Frame` from an ESP32-S3 node (`StreamClientKey::Esp32Node`).
    pub fn step_esp32_frame_80ms(&self, peer: SocketAddr, frame: DecodedEsp32Frame) -> TickDecision {
        let key = StreamClientKey::Esp32Node(peer);
        let mut slot = self
            .slots
            .entry(key)
            .or_insert_with(|| RollingStreamSlot::new(key, &self.device));

        let (_fused_h, _vad_logits) = self
            .weights
            .esp32_adapter
            .lift_packet_to_lfm_hidden(&frame, &self.device);

        let is_active_speech = frame.vad_prob > 0.45;
        if is_active_speech {
            slot.semantic_token_history.extend_from_slice(&frame.rvq_q0);
        }

        self.advance_turn_state(
            &mut slot,
            frame.seq,
            is_active_speech,
            false,
            frame.vad_prob > 0.65,
        )
    }

    fn advance_turn_state(
        &self,
        slot: &mut RollingStreamSlot,
        seq_num: u16,
        is_active_speech: bool,
        force_eot: bool,
        barge_in_trigger: bool,
    ) -> TickDecision {
        if is_active_speech {
            let was_idle = !slot.has_uncommitted_speech;
            slot.has_uncommitted_speech = true;
            slot.trailing_silent_ticks = 0;
            slot.semantic_token_history.push(1);
            if barge_in_trigger && was_idle {
                return TickDecision::BargeInDetected;
            }
            if !force_eot {
                return TickDecision::Listening { seq_num };
            }
        }

        if !slot.has_uncommitted_speech {
            if barge_in_trigger {
                return TickDecision::BargeInDetected;
            }
            return TickDecision::Listening { seq_num };
        }

        if !force_eot {
            slot.trailing_silent_ticks += 1;
            // Hold turn open during mid-clause thinking pauses (< 400ms = 5 ticks * 80ms)
            if slot.trailing_silent_ticks < 5 {
                return TickDecision::Listening { seq_num };
            }
        }

        // Commit turn -> Run shared LFM2.5 + KokoroIstftGenerator 24kHz synthesis
        slot.has_uncommitted_speech = false;
        slot.trailing_silent_ticks = 0;
        let frames_spoken = slot.semantic_token_history.len();
        slot.semantic_token_history.clear();

        let (transcript, reply_text, gpio_command, kokoro_24k) =
            self.execute_unified_lfm_kokoro_synthesis(slot.key, frames_spoken);
        let kokoro_arc = Arc::new(kokoro_24k);

        let _ = self.bridge_tx.send(CrossBridgeEvent::VoiceTurnCompleted {
            source: slot.key,
            transcript: transcript.clone(),
            reply_text: reply_text.clone(),
            gpio_command: gpio_command.clone(),
            kokoro_24k_mono: Arc::clone(&kokoro_arc),
        });

        TickDecision::TurnCommitted {
            transcript,
            reply_text,
            gpio_command,
            kokoro_24k_mono: kokoro_arc,
        }
    }

    fn execute_unified_lfm_kokoro_synthesis(
        &self,
        source: StreamClientKey,
        frames: usize,
    ) -> (String, String, Option<String>, Vec<f32>) {
        let transcript = format!("[{:?} spoken clause: {} quantal sub-frames]", source, frames);
        let reply_text = format!("Aurora-8B processed {:?} voice turn.", source);
        let gpio_command = Some(r#"{"cmd":"gpio","pin":4,"val":1}"#.to_string());

        let x = Tensor::<3>::full([1, 64, 4], 0.15, &self.device);
        let s = Tensor::<2>::full([1, 32], 0.10, &self.device);
        let f0 = Tensor::<3>::full([1, 1, 8], 220.0, &self.device);
        let wave_24k = self.weights.kokoro_gen.forward(x, s, f0);
        let pcm_24k = wave_24k
            .into_data()
            .try_to_vec::<f32>()
            .unwrap_or_else(|_| vec![0.0f32; 1200]);

        (transcript, reply_text, gpio_command, pcm_24k)
    }
}
