use std::{
    collections::VecDeque,
    io::{Read, Result as IoResult, Seek, SeekFrom},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex,
    },
};

use songbird::input::{Input, RawAdapter};
use symphonia::core::io::MediaSource;

pub const DISCORD_SAMPLE_RATE: u32 = 48_000;
pub const DISCORD_CHANNELS: usize = 2;

/// Shared lock-free-style handle allowing the Burn inference loop to push `24 kHz` mono PCM
/// from `KokoroIstftGenerator` or trigger instant `< 1 ms` `barge_in_flush()` into Songbird's stream.
#[derive(Clone)]
pub struct DuplexAudioSink {
    pcm_48k_stereo_ring: Arc<Mutex<VecDeque<f32>>>,
    is_speaking: Arc<AtomicBool>,
    pub barge_in_flushes: Arc<AtomicU64>,
    pub pushed_samples_48k: Arc<AtomicU64>,
}

impl Default for DuplexAudioSink {
    fn default() -> Self {
        Self::new()
    }
}

impl DuplexAudioSink {
    pub fn new() -> Self {
        Self {
            pcm_48k_stereo_ring: Arc::new(Mutex::new(VecDeque::with_capacity(48_000 * 4))),
            is_speaking: Arc::new(AtomicBool::new(false)),
            barge_in_flushes: Arc::new(AtomicU64::new(0)),
            pushed_samples_48k: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Upsamples `24 kHz` Mono `f32` from Burn's `KokoroIstftGenerator` -> `48 kHz` Stereo `f32`
    /// (`1:2` Hermite interpolation) and pushes to Songbird's persistent stream.
    pub fn push_kokoro_24k_mono(&self, samples_24k: &[f32]) {
        if samples_24k.is_empty() {
            return;
        }
        let stereo_48k = aurora_core::resampler::upsample_24k_mono_to_48k_stereo(samples_24k);
        let mut ring = self
            .pcm_48k_stereo_ring
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        ring.extend(stereo_48k.iter().copied());
        self.pushed_samples_48k
            .fetch_add(stereo_48k.len() as u64, Ordering::Relaxed);
        self.is_speaking.store(true, Ordering::Release);
    }

    /// Pushes pre-upsampled `48 kHz` interleaved stereo `f32` samples directly into the ring buffer.
    pub fn push_48k_stereo(&self, stereo_48k: &[f32]) {
        if stereo_48k.is_empty() {
            return;
        }
        let mut ring = self
            .pcm_48k_stereo_ring
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        ring.extend(stereo_48k.iter().copied());
        self.pushed_samples_48k
            .fetch_add(stereo_48k.len() as u64, Ordering::Relaxed);
        self.is_speaking.store(true, Ordering::Release);
    }

    /// Instant barge-in: clears all pending TTS audio within `< 1 ms` when Semantic VAD detects user speech.
    pub fn barge_in_flush(&self) {
        let mut ring = self
            .pcm_48k_stereo_ring
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        ring.clear();
        self.is_speaking.store(false, Ordering::Release);
        self.barge_in_flushes.fetch_add(1, Ordering::Relaxed);
    }

    pub fn is_bot_currently_speaking(&self) -> bool {
        self.is_speaking.load(Ordering::Acquire)
    }

    pub fn queued_samples(&self) -> usize {
        self.pcm_48k_stereo_ring
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len()
    }

    /// Wraps this sink into a Songbird `Input` ready for `call.play_only_input(...)`.
    pub fn to_songbird_input(&self) -> Input {
        let reader = ContinuousPcmSource {
            ring: Arc::clone(&self.pcm_48k_stereo_ring),
            is_speaking: Arc::clone(&self.is_speaking),
        };
        let adapter = RawAdapter::new(reader, DISCORD_SAMPLE_RATE, DISCORD_CHANNELS as u32);
        adapter.into()
    }

    /// Create a raw `ContinuousPcmSource` reader for unit testing.
    pub fn new_reader(&self) -> ContinuousPcmSource {
        ContinuousPcmSource {
            ring: Arc::clone(&self.pcm_48k_stereo_ring),
            is_speaking: Arc::clone(&self.is_speaking),
        }
    }
}

/// Implements `Read + Seek + MediaSource` emitting continuous `48 kHz` Stereo `f32le` PCM to Songbird.
/// When the TTS queue is empty, emits zero-amplitude silence so the track stays alive 24/7.
pub struct ContinuousPcmSource {
    ring: Arc<Mutex<VecDeque<f32>>>,
    is_speaking: Arc<AtomicBool>,
}

impl Read for ContinuousPcmSource {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<usize> {
        let max_floats = buf.len() / 4;
        let mut ring = self.ring.lock().unwrap_or_else(|e| e.into_inner());

        for i in 0..max_floats {
            let sample = ring.pop_front().unwrap_or(0.0);
            let bytes = sample.to_le_bytes();
            let off = i * 4;
            buf[off..off + 4].copy_from_slice(&bytes);
        }

        if ring.is_empty() {
            self.is_speaking.store(false, Ordering::Release);
        }

        Ok(max_floats * 4)
    }
}

impl Seek for ContinuousPcmSource {
    fn seek(&mut self, _pos: SeekFrom) -> IoResult<u64> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "Live duplex stream is unseekable",
        ))
    }
}

impl MediaSource for ContinuousPcmSource {
    fn is_seekable(&self) -> bool {
        false
    }

    fn byte_len(&self) -> Option<u64> {
        None
    }
}
