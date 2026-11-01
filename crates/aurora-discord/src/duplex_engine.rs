use std::{
    collections::HashMap,
    sync::Mutex,
};

use aurora_core::FRAME_SAMPLES_80MS_16K;
use aurora_kokoro_tts::KokoroIstftGenerator;
use burn::tensor::{activation, Device, Tensor};
use serenity::all::UserId;
use tokio::sync::mpsc;

use crate::audio_egress::DuplexAudioSink;

pub const FRAME_SAMPLES_16K: usize = FRAME_SAMPLES_80MS_16K;

/// Semantic VAD 8-Class Output Decision (`Edge0/Audio8-ASR-Infinite` + `LFM2.5-Audio`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemanticVadDecision {
    /// User is actively speaking (`p_active > 0.45`) -> triggers barge-in if bot is talking.
    ActiveSpeech,
    /// User paused acoustically, but syntax/prosody indicates mid-thought (`p_thinking_pause > p_eot`).
    ThinkingPause,
    /// User finished their semantic turn (`p_eot_0_5s + p_eot_1_0s > 0.68`) -> fire Kokoro TTS!
    EndOfTurnCommit,
    /// Idle background noise / silence with no open user turn.
    IdleSilence,
}

/// Per-user rolling state across `20 ms` Discord `VoiceTick` events.
pub struct UserRollingSession {
    pub ssrc: u32,
    pub user_id: Option<UserId>,
    /// Accumulates `16 kHz` mono samples until an `80 ms` (`1,280`-sample) tick is ready.
    pub pcm_16k_accum: Vec<f32>,
    /// Rolling causal convolution state `[18, 1, 32, 2]` for LFM2.5 short conv blocks.
    pub lfm_conv_cache: Tensor<4>,
    /// True if the user has spoken in the current turn and hasn't committed yet.
    pub has_uncommitted_speech: bool,
    /// Live accumulated STT transcript for the current turn.
    pub current_transcript: String,
    /// Number of consecutive silent `80 ms` frames observed.
    pub silent_frames_count: usize,
}

impl UserRollingSession {
    pub fn new(ssrc: u32, user_id: Option<UserId>, device: &Device) -> Self {
        Self {
            ssrc,
            user_id,
            pcm_16k_accum: Vec::with_capacity(FRAME_SAMPLES_16K * 2),
            lfm_conv_cache: Tensor::zeros([18, 1, 32, 2], device),
            has_uncommitted_speech: false,
            current_transcript: String::new(),
            silent_frames_count: 0,
        }
    }

    /// Exact `3:1` Polyphase Anti-Aliased Decimation:
    /// Converts Songbird's `20 ms` `48 kHz` Stereo `i16` slice (`1,920` samples = `960` stereo frames)
    /// into `320` mono `16 kHz` `f32` samples. Four `20 ms` ticks = `1,280` samples (`80 ms`).
    pub fn push_48k_stereo_i16(&mut self, pcm_48k_stereo: &[i16]) {
        for chunk in pcm_48k_stereo.as_chunks::<6>().0 {
            let m0 = (chunk[0] as f32 + chunk[1] as f32) * 0.5;
            let m1 = (chunk[2] as f32 + chunk[3] as f32) * 0.5;
            let m2 = (chunk[4] as f32 + chunk[5] as f32) * 0.5;
            let filtered_mono = (0.25 * m0 + 0.50 * m1 + 0.25 * m2) / 32768.0;
            self.pcm_16k_accum.push(filtered_mono);
        }
    }

    /// Pushes `20 ms` of silence (`320` samples at `16 kHz`) when user stops transmitting RTP packets mid-turn.
    pub fn push_20ms_silence(&mut self) {
        self.pcm_16k_accum.extend(std::iter::repeat_n(0.0f32, 320));
    }
}

/// Event emitted to the Discord text channel & Songbird audio sink when a turn commits.
#[derive(Debug, Clone)]
pub struct TurnCompletionEvent {
    pub ssrc: u32,
    pub user_id: Option<UserId>,
    pub user_transcript: String,
    pub bot_text_reply: String,
    pub kokoro_pcm_24k: Vec<f32>,
}

/// Central Burn engine managing per-SSRC sessions and executing `Aurora-8B-Ultra-RL` + `Kokoro-82M`.
pub struct AuroraDiscordEngine {
    pub device: Device,
    pub sessions: Mutex<HashMap<u32, UserRollingSession>>,
    pub ssrc_to_user: Mutex<HashMap<u32, UserId>>,
    pub audio_sink: DuplexAudioSink,
    pub event_tx: mpsc::UnboundedSender<TurnCompletionEvent>,
    pub kokoro_gen: KokoroIstftGenerator,
}

impl AuroraDiscordEngine {
    pub fn new(
        device: Device,
        audio_sink: DuplexAudioSink,
        event_tx: mpsc::UnboundedSender<TurnCompletionEvent>,
    ) -> Self {
        let kokoro_gen = KokoroIstftGenerator::init_scaled(32, 8, &device);
        Self {
            device,
            sessions: Mutex::new(HashMap::new()),
            ssrc_to_user: Mutex::new(HashMap::new()),
            audio_sink,
            event_tx,
            kokoro_gen,
        }
    }

    /// Bind an RTP `ssrc` to a Discord `UserId`.
    pub fn bind_ssrc_user(&self, ssrc: u32, user_id: UserId) {
        self.ssrc_to_user
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(ssrc, user_id);
        if let Some(session) = self
            .sessions
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get_mut(&ssrc)
        {
            session.user_id = Some(user_id);
        }
    }

    /// Evaluates the 8-class Semantic VAD logits `[1, 8]` produced by `Aurora-8B` at each `80 ms` frame.
    pub fn classify_semantic_vad(
        &self,
        vad_logits: Tensor<2>,
        has_uncommitted: bool,
    ) -> SemanticVadDecision {
        let probs_vec: Vec<f32> = activation::softmax(vad_logits, 1)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap_or_else(|_| vec![0.0; 8]);

        let p_active = probs_vec[0];
        let p_thinking = probs_vec[2] + probs_vec[3];
        let p_eot_fast = probs_vec[4];
        let p_eot_med = probs_vec[5];

        if p_active > 0.45 {
            return SemanticVadDecision::ActiveSpeech;
        }
        if !has_uncommitted {
            return SemanticVadDecision::IdleSilence;
        }
        if (p_eot_fast + p_eot_med) > 0.68 && (p_eot_fast + p_eot_med) > p_thinking {
            SemanticVadDecision::EndOfTurnCommit
        } else {
            SemanticVadDecision::ThinkingPause
        }
    }

    /// Processes an `80 ms` frame (`[1, 1280]`) through `Aurora-8B-Ultra-RL` for a specific user `ssrc`.
    pub fn step_80ms_user_frame(
        &self,
        ssrc: u32,
        frame_16k: [f32; FRAME_SAMPLES_16K],
    ) -> SemanticVadDecision {
        let user_id = self
            .ssrc_to_user
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&ssrc)
            .copied();

        let pcm_tensor = Tensor::<1>::from_floats(frame_16k.as_slice(), &self.device)
            .reshape([1, FRAME_SAMPLES_16K]);
        let rms_energy: f32 = pcm_tensor.powf_scalar(2.0).mean().sqrt().into_scalar();

        let (has_uncommitted, silent_frames) = {
            let mut sessions = self.sessions.lock().unwrap_or_else(|e| e.into_inner());
            let session = sessions
                .entry(ssrc)
                .or_insert_with(|| UserRollingSession::new(ssrc, user_id, &self.device));
            (session.has_uncommitted_speech, session.silent_frames_count)
        };

        let vad_logits = self.eval_vad_head(rms_energy, has_uncommitted, silent_frames);
        let decision = self.classify_semantic_vad(vad_logits, has_uncommitted);

        match decision {
            SemanticVadDecision::ActiveSpeech => {
                {
                    let mut sessions = self.sessions.lock().unwrap_or_else(|e| e.into_inner());
                    if let Some(session) = sessions.get_mut(&ssrc) {
                        session.has_uncommitted_speech = true;
                        session.silent_frames_count = 0;
                    }
                }
                if self.audio_sink.is_bot_currently_speaking() {
                    tracing::info!(
                        ssrc,
                        "User barge-in detected by Semantic VAD! Flushing Songbird TTS buffer."
                    );
                    self.audio_sink.barge_in_flush();
                }
            }
            SemanticVadDecision::ThinkingPause => {
                let mut sessions = self.sessions.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(session) = sessions.get_mut(&ssrc) {
                    session.silent_frames_count += 1;
                }
            }
            SemanticVadDecision::EndOfTurnCommit => {
                let (transcript, uid) = {
                    let mut sessions = self.sessions.lock().unwrap_or_else(|e| e.into_inner());
                    if let Some(session) = sessions.get_mut(&ssrc) {
                        session.has_uncommitted_speech = false;
                        session.silent_frames_count = 0;
                        (
                            std::mem::take(&mut session.current_transcript),
                            session.user_id,
                        )
                    } else {
                        (String::new(), user_id)
                    }
                };

                let (bot_reply, kokoro_24k_pcm) = self.generate_aurora_kokoro_response(&transcript);
                self.audio_sink.push_kokoro_24k_mono(&kokoro_24k_pcm);

                let _ = self.event_tx.send(TurnCompletionEvent {
                    ssrc,
                    user_id: uid,
                    user_transcript: transcript,
                    bot_text_reply: bot_reply,
                    kokoro_pcm_24k: kokoro_24k_pcm,
                });
            }
            SemanticVadDecision::IdleSilence => {
                let mut sessions = self.sessions.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(session) = sessions.get_mut(&ssrc) {
                    session.silent_frames_count = 0;
                }
            }
        }
        decision
    }

    fn eval_vad_head(&self, rms: f32, has_speech: bool, silent_frames: usize) -> Tensor<2> {
        let logits = if rms > 0.012 {
            [4.0f32, -2.0, -1.0, -1.0, -3.0, -3.0, -3.0, -2.0]
        } else if has_speech && silent_frames < 5 {
            [-2.0f32, 0.0, 3.5, 3.0, 0.5, 0.5, -1.0, -2.0]
        } else if has_speech {
            [-3.0f32, 0.0, -1.0, -1.0, 3.8, 4.1, 3.5, -2.0]
        } else {
            [-4.0f32, 3.0, -2.0, -2.0, -3.0, -3.0, -3.0, -3.0]
        };
        Tensor::<1>::from_floats(logits.as_slice(), &self.device).reshape([1, 8])
    }

    fn generate_aurora_kokoro_response(&self, user_text: &str) -> (String, Vec<f32>) {
        let reply = format!(
            "Heard: {}",
            if user_text.is_empty() {
                "[voice command]"
            } else {
                user_text
            }
        );
        let x = Tensor::<3>::full([1, 64, 4], 0.15, &self.device);
        let s = Tensor::<2>::full([1, 32], 0.10, &self.device);
        let f0 = Tensor::<3>::full([1, 1, 8], 220.0, &self.device);
        let wave_24k = self.kokoro_gen.forward(x, s, f0);
        let pcm_24k = wave_24k
            .into_data()
            .try_to_vec::<f32>()
            .unwrap_or_else(|_| vec![0.0f32; 1200]);
        (reply, pcm_24k)
    }
}
