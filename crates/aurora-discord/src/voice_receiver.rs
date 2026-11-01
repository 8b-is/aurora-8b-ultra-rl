use std::sync::Arc;

use serenity::{all::UserId, async_trait};
use songbird::{
    model::payload::Speaking,
    Event, EventContext, EventHandler as VoiceEventHandler,
};

use crate::duplex_engine::{AuroraDiscordEngine, UserRollingSession, FRAME_SAMPLES_16K};

/// Songbird `VoiceEventHandler` subscribing to both `SpeakingStateUpdate` and `VoiceTick`.
/// Crucially iterates over both `tick.speaking` AND `tick.silent` SSRCs so the `80 ms`
/// Semantic VAD horizon clock progresses when a speaker pauses mid-thought or finishes their turn.
#[derive(Clone)]
pub struct AuroraVoiceReceiver {
    pub engine: Arc<AuroraDiscordEngine>,
}

#[async_trait]
impl VoiceEventHandler for AuroraVoiceReceiver {
    async fn act(&self, ctx: &EventContext<'_>) -> Option<Event> {
        match ctx {
            EventContext::SpeakingStateUpdate(Speaking {
                ssrc,
                user_id: Some(uid),
                ..
            }) => {
                let serenity_uid = UserId::new(uid.0);
                self.engine.bind_ssrc_user(*ssrc, serenity_uid);
            }
            EventContext::VoiceTick(tick) => {
                // A. Demux actively speaking users (48 kHz Stereo i16 -> 16 kHz Mono f32)
                for (&ssrc, data) in &tick.speaking {
                    if let Some(pcm_48k_stereo) = &data.decoded_voice {
                        let mut ready_frames: Vec<[f32; FRAME_SAMPLES_16K]> = Vec::new();
                        {
                            let uid = self
                                .engine
                                .ssrc_to_user
                                .lock()
                                .unwrap_or_else(|e| e.into_inner())
                                .get(&ssrc)
                                .copied();
                            let mut sessions = self
                                .engine
                                .sessions
                                .lock()
                                .unwrap_or_else(|e| e.into_inner());
                            let session = sessions.entry(ssrc).or_insert_with(|| {
                                UserRollingSession::new(ssrc, uid, &self.engine.device)
                            });

                            session.push_48k_stereo_i16(pcm_48k_stereo);

                            while session.pcm_16k_accum.len() >= FRAME_SAMPLES_16K {
                                let mut frame = [0.0f32; FRAME_SAMPLES_16K];
                                for (i, sample) in
                                    session.pcm_16k_accum.drain(..FRAME_SAMPLES_16K).enumerate()
                                {
                                    frame[i] = sample;
                                }
                                ready_frames.push(frame);
                            }
                        }

                        for frame in ready_frames {
                            let _ = self.engine.step_80ms_user_frame(ssrc, frame);
                        }
                    }
                }

                // B. Advance Semantic VAD horizon clock for users who paused transmitting RTP packets
                for &ssrc in &tick.silent {
                    let mut ready_frames: Vec<[f32; FRAME_SAMPLES_16K]> = Vec::new();
                    {
                        let mut sessions = self
                            .engine
                            .sessions
                            .lock()
                            .unwrap_or_else(|e| e.into_inner());
                        if let Some(session) = sessions.get_mut(&ssrc) {
                            if session.has_uncommitted_speech {
                                session.push_20ms_silence();
                                while session.pcm_16k_accum.len() >= FRAME_SAMPLES_16K {
                                    let mut frame = [0.0f32; FRAME_SAMPLES_16K];
                                    for (i, sample) in session
                                        .pcm_16k_accum
                                        .drain(..FRAME_SAMPLES_16K)
                                        .enumerate()
                                    {
                                        frame[i] = sample;
                                    }
                                    ready_frames.push(frame);
                                }
                            }
                        }
                    }

                    for frame in ready_frames {
                        let _ = self.engine.step_80ms_user_frame(ssrc, frame);
                    }
                }
            }
            _ => {}
        }
        None
    }
}
