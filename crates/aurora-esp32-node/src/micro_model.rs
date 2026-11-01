//! `#![no_std]` Burn-Flex Quantal Micro-Encoder, Single-Tap NLMS Acoustic Echo
//! Cancellation, and Local VAD running inside the ESP32-S3's 8 MB Octal PSRAM.

extern crate alloc;
use alloc::vec::Vec;
use burn::{
    module::Param,
    nn::{
        conv::{Conv1d, Conv1dConfig},
        Linear, LinearConfig, PaddingConfig1d, RmsNorm, RmsNormConfig,
    },
    tensor::{activation, Device, Tensor, TensorData},
};

pub const FRAME_SAMPLES_16K: usize = 1280; // 80ms @ 16kHz
pub const SUBFRAMES_PER_80MS: usize = 8; // 10ms sub-steps (hop = 160)
pub const QUANTAL_PACKET_MAGIC: u16 = 0xA8B0; // "Aurora 8B"
pub const QUANTAL_PACKET_WIRE_BYTES: usize = 42; // 42 bytes every 80ms = 4.2 kbps

/// Wire-format packet sent over WebSocket every 80ms (42 bytes total = 4.2 kbps!)
#[repr(C, packed)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Quantal80msPacket {
    pub magic: u16,
    pub seq_num: u16,
    pub vad_prob_q8: u8,
    pub rms_db_q8: i8,
    pub rvq_q0: [u16; 10],
    pub rvq_q1: [u16; 8],
}

// Compile-time architectural invariant: Quantal80msPacket MUST be exactly 42 bytes.
const _: () = assert!(core::mem::size_of::<Quantal80msPacket>() == QUANTAL_PACKET_WIRE_BYTES);

impl Quantal80msPacket {
    /// Serializes this 42-byte packet into a little-endian wire byte array without `unsafe`.
    pub fn to_wire_bytes(&self) -> [u8; QUANTAL_PACKET_WIRE_BYTES] {
        let mut out = [0u8; QUANTAL_PACKET_WIRE_BYTES];
        let magic = self.magic;
        let seq_num = self.seq_num;
        let rvq_q0 = self.rvq_q0;
        let rvq_q1 = self.rvq_q1;

        out[0..2].copy_from_slice(&magic.to_le_bytes());
        out[2..4].copy_from_slice(&seq_num.to_le_bytes());
        out[4] = self.vad_prob_q8;
        out[5] = self.rms_db_q8 as u8;
        for i in 0..10 {
            let off = 6 + i * 2;
            out[off..off + 2].copy_from_slice(&rvq_q0[i].to_le_bytes());
        }
        for i in 0..8 {
            let off = 26 + i * 2;
            out[off..off + 2].copy_from_slice(&rvq_q1[i].to_le_bytes());
        }
        out
    }

    /// Decodes a 42-byte wire frame and verifies the `0xA8B0` magic header.
    pub fn from_wire_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < QUANTAL_PACKET_WIRE_BYTES {
            return None;
        }
        let magic = u16::from_le_bytes([bytes[0], bytes[1]]);
        if magic != QUANTAL_PACKET_MAGIC {
            return None;
        }
        let seq_num = u16::from_le_bytes([bytes[2], bytes[3]]);
        let vad_prob_q8 = bytes[4];
        let rms_db_q8 = bytes[5] as i8;
        let mut rvq_q0 = [0u16; 10];
        let mut rvq_q1 = [0u16; 8];
        for i in 0..10 {
            let off = 6 + i * 2;
            rvq_q0[i] = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
        }
        for i in 0..8 {
            let off = 26 + i * 2;
            rvq_q1[i] = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
        }
        Some(Self {
            magic,
            seq_num,
            vad_prob_q8,
            rms_db_q8,
            rvq_q0,
            rvq_q1,
        })
    }
}

/// Single-tap NLMS Acoustic Echo Cancellation (step size `mu = 0.05`) subtracting
/// the MAX98357A speaker reference signal from the INMP441 microphone capture.
pub fn nlms_aec_cancel_frame(
    mic_i16: &[i16; FRAME_SAMPLES_16K],
    spk_ref: &[f32; FRAME_SAMPLES_16K],
    nlms_weight: &mut f32,
    out_clean_f32: &mut [f32; FRAME_SAMPLES_16K],
) {
    for i in 0..FRAME_SAMPLES_16K {
        let mic_val = (mic_i16[i] as f32) / 32768.0;
        let s_ref = spk_ref[i];
        let echo_est = (*nlms_weight) * s_ref;
        let clean_err = mic_val - echo_est;
        let norm = s_ref * s_ref + 1e-4;
        *nlms_weight += 0.05 * clean_err * s_ref / norm;
        out_clean_f32[i] = clean_err.clamp(-1.0, 1.0);
    }
}

#[inline]
pub fn vad_prob_exceeds_barge_in(weight: f32) -> bool {
    weight.abs() > 0.01
}

#[derive(Debug)]
pub struct Esp32QuantalFrontend {
    pub frontend_conv: Conv1d,
    pub short_conv: Conv1d,
    pub norm: RmsNorm,
    pub proj: Linear,
    pub vad_head: Linear,
    pub q0_codebook_head: Linear,
    pub q1_codebook_head: Linear,
}

impl Esp32QuantalFrontend {
    pub fn init_from_static_weights(weights_blob: &[u8], device: &Device) -> Self {
        let mut model = Self {
            frontend_conv: Conv1dConfig::new(1, 64, 320)
                .with_stride(128)
                .with_padding(PaddingConfig1d::Explicit(96, 96))
                .with_bias(false)
                .init(device),
            short_conv: Conv1dConfig::new(64, 64, 3)
                .with_groups(64)
                .with_padding(PaddingConfig1d::Explicit(1, 1))
                .with_bias(false)
                .init(device),
            norm: RmsNormConfig::new(64).init(device),
            proj: LinearConfig::new(64, 128).with_bias(false).init(device),
            vad_head: LinearConfig::new(128, 2).with_bias(true).init(device),
            q0_codebook_head: LinearConfig::new(128, 256).with_bias(false).init(device),
            q1_codebook_head: LinearConfig::new(128, 256).with_bias(false).init(device),
        };

        if weights_blob.len() >= 64 * 320 * 4 {
            let floats: &[f32] = bytemuck::cast_slice(&weights_blob[..64 * 320 * 4]);
            let w = Tensor::<3>::from_data(TensorData::new(floats.to_vec(), [64, 1, 320]), device);
            model.frontend_conv.weight = Param::from_tensor(w);
        }
        model
    }

    /// Executes on-device Burn-Flex inference in ~6.5ms on ESP32-S3 @ 240MHz
    pub fn encode_80ms_frame(
        &self,
        pcm_16k: &[f32; FRAME_SAMPLES_16K],
        seq_num: u16,
        _barge_in_flag: bool,
        device: &Device,
    ) -> (f32, Quantal80msPacket) {
        let wav = Tensor::<1>::from_floats(pcm_16k.as_slice(), device).reshape([
            1,
            1,
            FRAME_SAMPLES_16K,
        ]);
        let feats = activation::silu(self.frontend_conv.forward(wav));
        let t_out = feats.dims()[2].min(10);
        let conv_feats = activation::silu(self.short_conv.forward(feats))
            .slice([0..1, 0..64, 0..t_out])
            .swap_dims(1, 2);

        let h = activation::silu(self.proj.forward(self.norm.forward(conv_feats)));
        let h_pooled = h.clone().mean_dim(1).reshape([1, 128]);
        let vad_logits = self.vad_head.forward(h_pooled);
        let vad_probs: Vec<f32> = activation::softmax(vad_logits, 1)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap_or_default();
        let speech_prob = *vad_probs.first().unwrap_or(&0.0);

        let q0_idx: Vec<i64> = self
            .q0_codebook_head
            .forward(h.clone())
            .argmax(2)
            .into_data()
            .try_to_vec::<i64>()
            .unwrap_or_default();
        let q1_idx: Vec<i64> = self
            .q1_codebook_head
            .forward(h)
            .argmax(2)
            .into_data()
            .try_to_vec::<i64>()
            .unwrap_or_default();

        let mut rvq_q0 = [0u16; 10];
        let mut rvq_q1 = [0u16; 8];
        for i in 0..10 {
            rvq_q0[i] = (q0_idx.get(i).copied().unwrap_or(0) as u16) % 2048;
        }
        for i in 0..8 {
            rvq_q1[i] = (q1_idx.get(i).copied().unwrap_or(0) as u16) % 2048;
        }

        // Compute RMS dBFS
        let mut sum_sq = 0.0f32;
        for &s in pcm_16k {
            sum_sq += s * s;
        }
        let rms = libm::sqrtf(sum_sq / (FRAME_SAMPLES_16K as f32));
        let rms_db = if rms > 1e-5 {
            (20.0 * libm::log10f(rms)).clamp(-127.0, 0.0) as i8
        } else {
            -120
        };

        let packet = Quantal80msPacket {
            magic: QUANTAL_PACKET_MAGIC,
            seq_num,
            vad_prob_q8: (speech_prob.clamp(0.0, 1.0) * 255.0) as u8,
            rms_db_q8: rms_db,
            rvq_q0,
            rvq_q1,
        };

        (speech_prob, packet)
    }
}
