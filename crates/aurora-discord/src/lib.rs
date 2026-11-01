pub mod audio_egress;
pub mod duplex_engine;
pub mod voice_receiver;

pub use audio_egress::{ContinuousPcmSource, DuplexAudioSink, DISCORD_CHANNELS, DISCORD_SAMPLE_RATE};
pub use duplex_engine::{
    AuroraDiscordEngine, SemanticVadDecision, TurnCompletionEvent, UserRollingSession,
    FRAME_SAMPLES_16K,
};
pub use voice_receiver::AuroraVoiceReceiver;

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Device;
    use tokio::sync::mpsc;

    #[test]
    fn test_duplex_audio_sink_barge_in_flush_under_1ms() {
        let sink = DuplexAudioSink::new();
        sink.push_kokoro_24k_mono(&vec![0.4f32; 2400]);
        assert!(sink.is_bot_currently_speaking());
        assert_eq!(sink.queued_samples(), 2400 * 4);

        sink.barge_in_flush();
        assert!(!sink.is_bot_currently_speaking());
        assert_eq!(sink.queued_samples(), 0);
    }

    #[test]
    fn test_8_class_semantic_vad_thinking_pause_vs_eot_commit() {
        let device = Device::default();
        let sink = DuplexAudioSink::new();
        let (tx, mut rx) = mpsc::unbounded_channel();
        let engine = AuroraDiscordEngine::new(device, sink, tx);

        // 1. Active speech frame (80 ms)
        let active_frame = [0.25f32; FRAME_SAMPLES_16K];
        let d1 = engine.step_80ms_user_frame(777, active_frame);
        assert_eq!(d1, SemanticVadDecision::ActiveSpeech);

        // 2. First 5 silent 80ms frames (< 400ms) -> ThinkingPause (holds turn!)
        let silent_frame = [0.0f32; FRAME_SAMPLES_16K];
        for _ in 0..5 {
            let d_pause = engine.step_80ms_user_frame(777, silent_frame);
            assert_eq!(d_pause, SemanticVadDecision::ThinkingPause);
        }

        // 3. 6th silent 80ms frame (>= 480ms) -> EndOfTurnCommit (triggers Kokoro-82M TTS!)
        let d_commit = engine.step_80ms_user_frame(777, silent_frame);
        assert_eq!(d_commit, SemanticVadDecision::EndOfTurnCommit);

        let evt = rx.try_recv().expect("TurnCompletionEvent must be emitted");
        assert_eq!(evt.ssrc, 777);
        assert!(!evt.kokoro_pcm_24k.is_empty());
    }
}
