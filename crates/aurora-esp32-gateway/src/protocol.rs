pub use aurora_core::{
    resampler::{
        chunk_esp32_ws_pcm_frames, resample_24k_to_16k_i16, ESP32_WS_MAX_CHUNK_BYTES,
        ESP32_WS_MIN_CHUNK_BYTES,
    },
    DecodedEsp32Frame, PacketDecodeError, Quantal80msPacket, QUANTAL_MAGIC_A8B0,
    QUANTAL_PACKET_BYTES,
};

/// Downsamples `24 kHz` mono `f32` (`KokoroIstftGenerator` output) to `16 kHz` `i16`
/// and slices it into RFC-6455 binary WebSocket payloads strictly within `128..=960` bytes.
pub fn kokoro_24k_to_esp32_ws_chunks(pcm_24k: &[f32]) -> Vec<Vec<u8>> {
    let pcm_16k = resample_24k_to_16k_i16(pcm_24k);
    chunk_esp32_ws_pcm_frames(&pcm_16k)
}
