#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod resampler;

/// Magic header (`0xA8B0` = Aurora-8B-0) identifying a 42-byte `Quantal80msPacket`.
pub const QUANTAL_MAGIC_A8B0: u16 = 0xA8B0;

/// Exact byte length of `Quantal80msPacket` on the wire (`42` bytes = `4.2 kbps` at `12.5 Hz`).
pub const QUANTAL_PACKET_BYTES: usize = 42;

/// Number of 16 kHz mono samples per 80 ms quantal frame (`16,000 * 0.080 = 1,280`).
pub const FRAME_SAMPLES_80MS_16K: usize = 1280;

/// Number of NLMS adaptive filter taps (`64` taps = `4 ms` echo tail at `16 kHz`).
pub const AEC_TAPS: usize = 64;

/// 42-byte wire packet transmitted every `80 ms` (`12.5 Hz` audio clock, `4.2 kbps` payload)
/// from the ESP32-S3 node to the `Aurora-8B-Ultra-RL` server over WebSocket.
#[repr(C, packed)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Quantal80msPacket {
    /// Magic header: `0xA8B0` (Aurora-8B-0) in little-endian.
    pub magic: u16,
    /// Monotonic 80ms frame sequence counter.
    pub seq: u16,
    /// Local micro-VAD speech probability (`0..=255` mapped to `0.0..=1.0`).
    pub vad_prob_q8: u8,
    /// Pre-AEC microphone RMS energy in dBFS (`-128..=0`).
    pub rms_db_q8: i8,
    /// Coarse semantic-acoustic RVQ indices (`10` sub-frames x `u16` in `0..2048`).
    pub rvq_q0: [u16; 10],
    /// Fine residual RVQ indices (`8` sub-frames x `u16` in `0..2048`).
    pub rvq_q1: [u16; 8],
}

// Compile-time architectural invariant: Quantal80msPacket MUST be exactly 42 bytes.
const _: () = assert!(core::mem::size_of::<Quantal80msPacket>() == QUANTAL_PACKET_BYTES);

/// Validated, host-aligned representation of an incoming 80ms ESP32 quantal frame.
#[derive(Clone, Debug, PartialEq)]
pub struct DecodedEsp32Frame {
    pub seq: u16,
    pub vad_prob: f32,
    pub rms_db: f32,
    pub rvq_q0: [u16; 10],
    pub rvq_q1: [u16; 8],
}

/// Wire-decoding errors for `Quantal80msPacket`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PacketDecodeError {
    InvalidLength(usize),
    InvalidMagic(u16),
    CodebookIndexOutOfBounds { codebook: u8, index: u16 },
}

impl Quantal80msPacket {
    /// Construct a new `Quantal80msPacket` with little-endian fields.
    pub fn new(
        seq: u16,
        vad_prob_q8: u8,
        rms_db_q8: i8,
        rvq_q0: [u16; 10],
        rvq_q1: [u16; 8],
    ) -> Self {
        Self {
            magic: QUANTAL_MAGIC_A8B0.to_le(),
            seq: seq.to_le(),
            vad_prob_q8,
            rms_db_q8,
            rvq_q0: rvq_q0.map(u16::to_le),
            rvq_q1: rvq_q1.map(u16::to_le),
        }
    }

    /// Zero-copy view of the 42-byte packet as a byte slice.
    #[inline]
    pub fn as_bytes(&self) -> &[u8; QUANTAL_PACKET_BYTES] {
        // SAFETY: `Quantal80msPacket` is `#[repr(C, packed)]` of size 42 with no padding bytes.
        unsafe { &*(self as *const Self as *const [u8; QUANTAL_PACKET_BYTES]) }
    }

    /// Decode and validate a 42-byte binary WebSocket payload into a `DecodedEsp32Frame`.
    pub fn decode_from_slice(bytes: &[u8]) -> Result<DecodedEsp32Frame, PacketDecodeError> {
        if bytes.len() != QUANTAL_PACKET_BYTES {
            return Err(PacketDecodeError::InvalidLength(bytes.len()));
        }

        let magic = u16::from_le_bytes([bytes[0], bytes[1]]);
        if magic != QUANTAL_MAGIC_A8B0 {
            return Err(PacketDecodeError::InvalidMagic(magic));
        }

        let seq = u16::from_le_bytes([bytes[2], bytes[3]]);
        let vad_prob = (bytes[4] as f32) / 255.0;
        let rms_db = (bytes[5] as i8) as f32;

        let mut rvq_q0 = [0u16; 10];
        for (i, slot) in rvq_q0.iter_mut().enumerate() {
            let offset = 6 + i * 2;
            let code = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
            if code >= 2048 {
                return Err(PacketDecodeError::CodebookIndexOutOfBounds {
                    codebook: 0,
                    index: code,
                });
            }
            *slot = code;
        }

        let mut rvq_q1 = [0u16; 8];
        for (i, slot) in rvq_q1.iter_mut().enumerate() {
            let offset = 26 + i * 2;
            let code = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
            if code >= 2048 {
                return Err(PacketDecodeError::CodebookIndexOutOfBounds {
                    codebook: 1,
                    index: code,
                });
            }
            *slot = code;
        }

        Ok(DecodedEsp32Frame {
            seq,
            vad_prob,
            rms_db,
            rvq_q0,
            rvq_q1,
        })
    }
}

/// 8-Class Multi-Horizon Semantic VAD classes predicted by `Aurora-8B-Ultra-RL` at 12.5 Hz.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SemanticVadClass {
    /// `[0]` Background noise or silence (`< -45 dBFS`).
    Silence = 0,
    /// `[1]` Active lexical speech token stream in progress.
    SpeechActive = 1,
    /// `[2]` Mid-sentence hesitation or thinking pause (`< 400 ms` / `< 5` ticks); hold turn.
    ThinkingPause = 2,
    /// `[3]` Early end-of-turn horizon (`250 ms` lookahead).
    EndOfTurn025s = 3,
    /// `[4]` Standard end-of-turn horizon (`500 ms` lookahead).
    EndOfTurn05s = 4,
    /// `[5]` High-confidence end-of-turn horizon (`1000 ms` lookahead).
    EndOfTurn10s = 5,
    /// `[6]` Short listener backchannel ("uh-huh", "yeah") that should not barge-in.
    Backchannel = 6,
    /// `[7]` Intentional user interruption while TTS is active; triggers `<1ms` barge-in flush.
    BargeInInterrupt = 7,
}

impl SemanticVadClass {
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => Self::Silence,
            1 => Self::SpeechActive,
            2 => Self::ThinkingPause,
            3 => Self::EndOfTurn025s,
            4 => Self::EndOfTurn05s,
            5 => Self::EndOfTurn10s,
            6 => Self::Backchannel,
            _ => Self::BargeInInterrupt,
        }
    }
}

/// Normalized Least-Mean-Squares (NLMS) Acoustic Echo Canceller (`64` taps at `16 kHz`).
/// Subtracts the estimated speaker playback echo from the microphone capture in-place
/// so the user can barge-in while `MAX98357A` I2S speaker playback is active.
pub fn nlms_aec_cancel_frame(
    mic_in: &[i16; FRAME_SAMPLES_80MS_16K],
    spk_ref: &[i16; FRAME_SAMPLES_80MS_16K],
    weights: &mut [f32; AEC_TAPS],
    delay_line: &mut [f32; AEC_TAPS],
    clean_out: &mut [f32; FRAME_SAMPLES_80MS_16K],
) -> f32 {
    const MU: f32 = 0.08;
    const EPS: f32 = 1e-4;
    let mut energy_acc = 0.0f32;

    for n in 0..FRAME_SAMPLES_80MS_16K {
        delay_line.copy_within(0..AEC_TAPS - 1, 1);
        delay_line[0] = (spk_ref[n] as f32) / 32768.0;

        let mut echo_est = 0.0f32;
        let mut ref_power = EPS;
        for k in 0..AEC_TAPS {
            let x_k = delay_line[k];
            echo_est += weights[k] * x_k;
            ref_power += x_k * x_k;
        }

        let d = (mic_in[n] as f32) / 32768.0;
        let err = libm::fminf(libm::fmaxf(d - echo_est, -1.0), 1.0);
        clean_out[n] = err;
        energy_acc += err * err;

        let step = (MU * err) / ref_power;
        for k in 0..AEC_TAPS {
            weights[k] += step * delay_line[k];
        }
    }

    libm::sqrtf(energy_acc / (FRAME_SAMPLES_80MS_16K as f32))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantal_80ms_packet_wire_roundtrip() {
        assert_eq!(core::mem::size_of::<Quantal80msPacket>(), 42);
        let q0 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 2047];
        let q1 = [1, 2, 3, 4, 5, 6, 7, 1024];
        let pkt = Quantal80msPacket::new(42, 204, -18, q0, q1);
        let wire = pkt.as_bytes();
        assert_eq!(wire.len(), 42);
        assert_eq!(wire[0], 0xB0);
        assert_eq!(wire[1], 0xA8);

        let decoded = Quantal80msPacket::decode_from_slice(wire).expect("valid 42B packet");
        assert_eq!(decoded.seq, 42);
        assert!((decoded.vad_prob - (204.0 / 255.0)).abs() < 1e-5);
        assert_eq!(decoded.rms_db, -18.0);
        assert_eq!(decoded.rvq_q0, q0);
        assert_eq!(decoded.rvq_q1, q1);
    }

    #[test]
    fn test_quantal_packet_rejects_invalid_codebook_index() {
        let mut q0 = [0u16; 10];
        q0[3] = 2048; // Out of bounds for RVQ 0..2047
        let pkt = Quantal80msPacket::new(1, 128, -20, q0, [0u16; 8]);
        let res = Quantal80msPacket::decode_from_slice(pkt.as_bytes());
        assert_eq!(
            res,
            Err(PacketDecodeError::CodebookIndexOutOfBounds {
                codebook: 0,
                index: 2048
            })
        );
    }

    #[test]
    fn test_nlms_aec_attenuates_correlated_echo() {
        let mut spk_ref = [0i16; FRAME_SAMPLES_80MS_16K];
        let mut mic_in = [0i16; FRAME_SAMPLES_80MS_16K];
        for i in 0..FRAME_SAMPLES_80MS_16K {
            let s = (libm::sinf(i as f32 * 0.15) * 12000.0) as i16;
            spk_ref[i] = s;
            mic_in[i] = (s as f32 * 0.6) as i16;
        }
        let mut weights = [0.0f32; AEC_TAPS];
        let mut delay = [0.0f32; AEC_TAPS];
        let mut clean = [0.0f32; FRAME_SAMPLES_80MS_16K];

        let rms_pass1 = nlms_aec_cancel_frame(&mic_in, &spk_ref, &mut weights, &mut delay, &mut clean);
        let rms_pass2 = nlms_aec_cancel_frame(&mic_in, &spk_ref, &mut weights, &mut delay, &mut clean);
        assert!(rms_pass2 < rms_pass1, "NLMS filter must reduce echo residual energy");
    }
}
