use std::{fs::File, io::Write, path::Path};

use anyhow::Result;

/// Simulates ESP32-S3 `INMP441` I2S MEMS microphone acoustic transfer characteristics
/// (1st-order high-pass filter at $f_c \approx 38\text{ Hz}$), `DNS-Challenge 5` 3-tap early-reflection
/// Room Impulse Response (`RIR`), and `MAX98357A` speaker echo leakage.
pub fn augment_esp32_mems_and_rir(clean_16k: &[f32], echo_ref_16k: &[f32], echo_gain: f32) -> Vec<f32> {
    let n = clean_16k.len();
    let mut out = Vec::with_capacity(n);
    let mut prev_x = 0.0f32;
    let mut prev_y = 0.0f32;

    for i in 0..n {
        // 1. 3-tap room impulse response (direct path + 1.5ms & 4.0ms early reflections)
        let direct = clean_16k[i];
        let refl_1 = if i >= 24 { clean_16k[i - 24] * 0.18 } else { 0.0 };
        let refl_2 = if i >= 64 { clean_16k[i - 64] * 0.09 } else { 0.0 };
        let echo = echo_ref_16k.get(i).copied().unwrap_or(0.0) * echo_gain;

        let raw = direct + refl_1 + refl_2 + echo;

        // 2. INMP441 MEMS 1st-order high-pass filter: y[n] = x[n] - x[n-1] + 0.985 * y[n-1]
        let hp = raw - prev_x + 0.985 * prev_y;
        prev_x = raw;
        prev_y = hp;

        out.push(hp.clamp(-1.0, 1.0));
    }

    out
}

/// Pure-Rust 16-bit PCM RIFF/WAVE writer (`24,000 Hz` mono) for `KokoroIstftGenerator` verification.
pub fn write_pcm16_wav_24k(path: &Path, samples_24k: &[f32]) -> Result< usize > {
    let sample_rate: u32 = 24_000;
    let num_channels: u16 = 1;
    let bits_per_sample: u16 = 16;
    let byte_rate = sample_rate * u32::from(num_channels) * u32::from(bits_per_sample / 8);
    let block_align = num_channels * (bits_per_sample / 8);
    let data_bytes = (samples_24k.len() * 2) as u32;
    let riff_size = 36 + data_bytes;

    let mut file = File::create(path)?;
    file.write_all(b"RIFF")?;
    file.write_all(&riff_size.to_le_bytes())?;
    file.write_all(b"WAVEfmt ")?;
    file.write_all(&16u32.to_le_bytes())?; // PCM chunk size
    file.write_all(&1u16.to_le_bytes())?; // AudioFormat = 1 (PCM)
    file.write_all(&num_channels.to_le_bytes())?;
    file.write_all(&sample_rate.to_le_bytes())?;
    file.write_all(&byte_rate.to_le_bytes())?;
    file.write_all(&block_align.to_le_bytes())?;
    file.write_all(&bits_per_sample.to_le_bytes())?;
    file.write_all(b"data")?;
    file.write_all(&data_bytes.to_le_bytes())?;

    for &s in samples_24k {
        let q15 = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        file.write_all(&q15.to_le_bytes())?;
    }

    Ok(44 + samples_24k.len() * 2)
}
