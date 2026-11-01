use alloc::vec::Vec;

/// Exact byte ceiling for a 30ms (`480` samples) 16kHz `i16le` WebSocket frame (`960` bytes).
pub const ESP32_WS_MAX_CHUNK_BYTES: usize = 960;

/// Minimum byte length (`128` bytes = `64` samples) for a server-to-ESP32 binary WebSocket frame
/// so that RFC-6455 always uses the 4-byte extended payload header (`[0x82, 126, len_hi, len_lo]`).
pub const ESP32_WS_MIN_CHUNK_BYTES: usize = 128;

/// 7-tap anti-aliasing low-pass FIR filter ($f_c \approx 7.2\text{ kHz}$) for `3:1` and `3:2` decimation.
const FIR_LPF_7TAP: [f32; 7] = [0.045, 0.110, 0.215, 0.260, 0.215, 0.110, 0.045];

/// Downsample `48 kHz` interleaved Stereo `i16` (`[L0, R0, L1, R1, ...]`) to `16 kHz` Mono `f32` (`[-1.0, 1.0]`).
/// Exactly `3:1` decimation (`3,840` stereo samples = `7,680` `i16` values per `80ms` -> `1,280` mono `f32` samples).
pub fn downsample_48k_stereo_to_16k_mono_f32(pcm_48k_stereo: &[i16]) -> Vec<f32> {
    let mono_48k_len = pcm_48k_stereo.len() / 2;
    let mut mono_48k = Vec::with_capacity(mono_48k_len);
    for frame in pcm_48k_stereo.chunks_exact(2) {
        let mid = ((frame[0] as f32) + (frame[1] as f32)) * (0.5 / 32768.0);
        mono_48k.push(mid);
    }

    let out_len = mono_48k_len / 3;
    let mut out_16k = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let center = i * 3;
        let mut acc = 0.0f32;
        for (tap_idx, &coeff) in FIR_LPF_7TAP.iter().enumerate() {
            let idx = (center + tap_idx).saturating_sub(3).min(mono_48k_len.saturating_sub(1));
            acc += mono_48k[idx] * coeff;
        }
        out_16k.push(acc.clamp(-1.0, 1.0));
    }
    out_16k
}

/// Downsample `48 kHz` Mono `i16` to `16 kHz` Mono `f32` (`3:1` anti-aliased FIR decimation).
pub fn downsample_48k_mono_to_16k_f32(pcm_48k_mono: &[i16]) -> Vec<f32> {
    let n = pcm_48k_mono.len();
    let out_len = n / 3;
    let mut out_16k = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let center = i * 3;
        let mut acc = 0.0f32;
        for (tap_idx, &coeff) in FIR_LPF_7TAP.iter().enumerate() {
            let idx = (center + tap_idx).saturating_sub(3).min(n.saturating_sub(1));
            acc += (pcm_48k_mono[idx] as f32 / 32768.0) * coeff;
        }
        out_16k.push(acc.clamp(-1.0, 1.0));
    }
    out_16k
}

/// Polyphase `3:2` resampler converting `24 kHz` mono `f32` (`Kokoro-82M` native output)
/// to `16 kHz` mono `i16` for ESP32-S3 `MAX98357A` I2S DMA playback.
/// Every `3` input samples (`24 kHz`) produce `2` output samples (`16 kHz`).
pub fn resample_24k_to_16k_i16(pcm_24k: &[f32]) -> Vec<i16> {
    if pcm_24k.is_empty() {
        return Vec::new();
    }
    let out_len = (pcm_24k.len() * 2) / 3;
    let mut out_16k = Vec::with_capacity(out_len);
    let last = pcm_24k.len() - 1;

    for m in 0..out_len {
        // Exact rational source coordinate: src_pos = m * 1.5
        let base = (m * 3) / 2;
        let is_half = (m & 1) == 1;
        let sample = if !is_half {
            // Integer phase: 3-tap symmetric smoothing `[0.15, 0.70, 0.15]`
            let s_prev = pcm_24k[base.saturating_sub(1)];
            let s_curr = pcm_24k[base.min(last)];
            let s_next = pcm_24k[(base + 1).min(last)];
            0.15 * s_prev + 0.70 * s_curr + 0.15 * s_next
        } else {
            // Half-sample phase (base + 0.5): 4-tap polyphase FIR `[-0.0625, 0.5625, 0.5625, -0.0625]`
            let s0 = pcm_24k[base.saturating_sub(1)];
            let s1 = pcm_24k[base.min(last)];
            let s2 = pcm_24k[(base + 1).min(last)];
            let s3 = pcm_24k[(base + 2).min(last)];
            -0.0625 * (s0 + s3) + 0.5625 * (s1 + s2)
        };
        let q15 = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
        out_16k.push(q15);
    }

    out_16k
}

/// Upsample `24 kHz` mono `f32` (`Kokoro-82M` output) to `48 kHz` interleaved Stereo `f32` (`1:2`)
/// using 4-point Hermite cubic interpolation for Songbird Discord voice egress.
pub fn upsample_24k_mono_to_48k_stereo(pcm_24k: &[f32]) -> Vec<f32> {
    let n = pcm_24k.len();
    if n == 0 {
        return Vec::new();
    }
    let mut stereo_48k = Vec::with_capacity(n * 4);
    for i in 0..n {
        let y0 = pcm_24k[i.saturating_sub(1)];
        let y1 = pcm_24k[i];
        let y2 = pcm_24k[(i + 1).min(n - 1)];
        let y3 = pcm_24k[(i + 2).min(n - 1)];

        // Even sample (t = 0.0)
        let s_even = y1.clamp(-1.0, 1.0);
        stereo_48k.push(s_even);
        stereo_48k.push(s_even);

        // Odd sample (t = 0.5) via 4-point cubic interpolation
        let s_odd = (-0.0625 * (y0 + y3) + 0.5625 * (y1 + y2)).clamp(-1.0, 1.0);
        stereo_48k.push(s_odd);
        stereo_48k.push(s_odd);
    }
    stereo_48k
}

/// Package `16 kHz` `i16` PCM samples into little-endian byte chunks strictly within `128..=960` bytes
/// (`30ms` = `480` samples = `960` bytes per full chunk, zero-padded to `>= 128` bytes on the tail)
/// so that every RFC-6455 binary WebSocket frame has a deterministic 4-byte header (`[0x82, 126, hi, lo]`).
pub fn chunk_esp32_ws_pcm_frames(pcm_16k: &[i16]) -> Vec<Vec<u8>> {
    if pcm_16k.is_empty() {
        return Vec::new();
    }
    let samples_per_chunk = ESP32_WS_MAX_CHUNK_BYTES / 2; // 480 samples (30ms)
    let mut frames = Vec::new();

    for chunk in pcm_16k.chunks(samples_per_chunk) {
        let mut buf = Vec::with_capacity(ESP32_WS_MAX_CHUNK_BYTES);
        for &s in chunk {
            buf.extend_from_slice(&s.to_le_bytes());
        }
        if buf.len() < ESP32_WS_MIN_CHUNK_BYTES {
            buf.resize(ESP32_WS_MIN_CHUNK_BYTES, 0u8);
        }
        frames.push(buf);
    }

    frames
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_downsample_48k_stereo_to_16k_mono_exact_ratio() {
        // 80ms at 48kHz stereo = 3,840 stereo frames = 7,680 i16 samples
        let stereo_48k = alloc::vec![16000i16; 7680];
        let mono_16k = downsample_48k_stereo_to_16k_mono_f32(&stereo_48k);
        assert_eq!(mono_16k.len(), 1280);
        assert!((mono_16k[640] - (16000.0 / 32768.0)).abs() < 1e-2);
    }

    #[test]
    fn test_resample_24k_to_16k_exact_ratio() {
        // 80ms at 24kHz = 1,920 samples -> 80ms at 16kHz = 1,280 samples
        let pcm_24k = alloc::vec![0.5f32; 1920];
        let pcm_16k = resample_24k_to_16k_i16(&pcm_24k);
        assert_eq!(pcm_16k.len(), 1280);
        assert!((pcm_16k[500] - 16383).abs() <= 2);
    }

    #[test]
    fn test_esp32_ws_chunker_enforces_128_to_960_byte_invariant() {
        // 500 samples = 480 samples (960B) + 20 tail samples (40B -> padded to 128B)
        let samples = alloc::vec![1024i16; 500];
        let chunks = chunk_esp32_ws_pcm_frames(&samples);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].len(), 960);
        assert_eq!(chunks[1].len(), 128);
    }
}
