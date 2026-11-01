pub mod shared_runtime;

pub use shared_runtime::{
    AuroraSharedWeights, CrossBridgeEvent, RollingStreamSlot, SharedAuroraRuntime, StreamClientKey,
    TickDecision,
};

#[cfg(test)]
mod tests {
    use super::*;
    use aurora_core::{Quantal80msPacket, FRAME_SAMPLES_80MS_16K};
    use burn::tensor::Device;
    use std::net::SocketAddr;

    #[test]
    fn test_shared_aurora_runtime_concurrent_discord_and_esp32_turns() {
        let device = Device::default();
        let runtime = SharedAuroraRuntime::new(device);
        let mut bridge_rx = runtime.bridge_tx.subscribe();

        // 1. ESP32 node sends speech frame -> barge-in detected when vad_prob > 0.65 on idle
        let peer: SocketAddr = "10.0.0.42:8080".parse().unwrap();
        let pkt_barge = Quantal80msPacket::new(1, 230, -12, [15; 10], [8; 8]);
        let frame_barge = Quantal80msPacket::decode_from_slice(pkt_barge.as_bytes()).unwrap();
        let d_barge = runtime.step_esp32_frame_80ms(peer, frame_barge);
        assert!(matches!(d_barge, TickDecision::BargeInDetected));

        // 2. Discord SSRC 999 speaks for 1 frame, pauses 4 frames (<400ms thinking), commits on 5th silent frame
        let speech_pcm = [0.22f32; FRAME_SAMPLES_80MS_16K];
        let d_speak = runtime.step_discord_pcm_80ms(999, speech_pcm, false);
        assert!(matches!(d_speak, TickDecision::Listening { .. }));

        let silence_pcm = [0.0f32; FRAME_SAMPLES_80MS_16K];
        for _ in 0..4 {
            let d_wait = runtime.step_discord_pcm_80ms(999, silence_pcm, false);
            assert!(matches!(d_wait, TickDecision::Listening { .. }));
        }

        let d_commit = runtime.step_discord_pcm_80ms(999, silence_pcm, false);
        assert!(matches!(d_commit, TickDecision::TurnCommitted { .. }));

        let bridge_evt = bridge_rx
            .try_recv()
            .expect("CrossBridgeEvent must broadcast across Discord & ESP32");
        match bridge_evt {
            CrossBridgeEvent::VoiceTurnCompleted {
                source,
                kokoro_24k_mono,
                ..
            } => {
                assert_eq!(source, StreamClientKey::DiscordSsrc(999));
                assert!(!kokoro_24k_mono.is_empty());
            }
            other => panic!("Unexpected event: {other:?}"),
        }
    }
}
