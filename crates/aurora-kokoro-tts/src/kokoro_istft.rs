use std::f32::consts::PI;

use burn::{
    module::Param,
    nn::{
        conv::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig},
        Linear, LinearConfig, PaddingConfig1d,
    },
    tensor::{
        activation,
        module::{conv1d, conv_transpose1d},
        ops::{ConvOptions, ConvTransposeOptions},
        Device, Tensor,
    },
};

pub const KOKORO_SAMPLE_RATE: u32 = 24_000;
pub const POST_N_FFT: usize = 20;
pub const POST_HOP_SIZE: usize = 5;
pub const POST_WIN_SIZE: usize = 20;
pub const FREQ_BINS: usize = POST_N_FFT / 2 + 1; // 11 bins
pub const HARMONIC_CHANNELS: usize = 9; // Fundamental F0 + 8 overtones

/// `AdaIN1d`: Adaptive Instance Normalization conditioned on a `style_dim` (`128`-d) speaker vector.
pub struct AdaIn1d {
    pub fc: Linear,
    pub channels: usize,
}

impl AdaIn1d {
    pub fn init(style_dim: usize, channels: usize, device: &Device) -> Self {
        Self {
            fc: LinearConfig::new(style_dim, channels * 2).init(device),
            channels,
        }
    }

    /// `x`: `[B, C, T]`, `s`: `[B, style_dim]`.
    pub fn forward(&self, x: Tensor<3>, s: Tensor<2>) -> Tensor<3> {
        let [b, c, _t] = x.dims();
        let mean = x.clone().mean_dim(2);
        let var = x.clone().var(2);
        let x_norm = (x - mean) / (var + 1e-5).sqrt();

        let h = self.fc.forward(s).reshape([b, c * 2, 1]);
        let chunks = h.chunk(2, 1);
        let (gamma, beta) = (chunks[0].clone(), chunks[1].clone());

        x_norm * (gamma + 1.0) + beta
    }
}

/// `Kokoro-82M` `AdaINResBlock1` with `Snake1D` periodic activation and dilated `Conv1d` (`d = [1, 3, 5]`).
pub struct AdaInResBlock1 {
    pub convs1: Vec<Conv1d>,
    pub convs2: Vec<Conv1d>,
    pub adain1: Vec<AdaIn1d>,
    pub adain2: Vec<AdaIn1d>,
    pub alpha1: Vec<Param<Tensor<3>>>,
    pub alpha2: Vec<Param<Tensor<3>>>,
}

impl AdaInResBlock1 {
    pub fn init(
        channels: usize,
        kernel_size: usize,
        dilations: [usize; 3],
        style_dim: usize,
        device: &Device,
    ) -> Self {
        let mut convs1 = Vec::with_capacity(3);
        let mut convs2 = Vec::with_capacity(3);
        let mut adain1 = Vec::with_capacity(3);
        let mut adain2 = Vec::with_capacity(3);
        let mut alpha1 = Vec::with_capacity(3);
        let mut alpha2 = Vec::with_capacity(3);

        for d in dilations {
            let pad1 = (kernel_size * d - d) / 2;
            let pad2 = (kernel_size - 1) / 2;
            convs1.push(
                Conv1dConfig::new(channels, channels, kernel_size)
                    .with_dilation(d)
                    .with_padding(PaddingConfig1d::Explicit(pad1, pad1))
                    .init(device),
            );
            convs2.push(
                Conv1dConfig::new(channels, channels, kernel_size)
                    .with_dilation(1)
                    .with_padding(PaddingConfig1d::Explicit(pad2, pad2))
                    .init(device),
            );
            adain1.push(AdaIn1d::init(style_dim, channels, device));
            adain2.push(AdaIn1d::init(style_dim, channels, device));
            alpha1.push(Param::from_tensor(Tensor::ones([1, channels, 1], device)));
            alpha2.push(Param::from_tensor(Tensor::ones([1, channels, 1], device)));
        }

        Self {
            convs1,
            convs2,
            adain1,
            adain2,
            alpha1,
            alpha2,
        }
    }

    /// `Snake1D` periodic activation: $x + \frac{1}{\alpha} \sin^2(\alpha x)$.
    #[inline]
    pub fn snake1d(x: Tensor<3>, alpha: Tensor<3>) -> Tensor<3> {
        let sin_ax = (x.clone() * alpha.clone()).sin();
        x + (sin_ax.powf_scalar(2.0) / (alpha + 1e-9))
    }

    pub fn forward(&self, mut x: Tensor<3>, s: Tensor<2>) -> Tensor<3> {
        for i in 0..3 {
            let mut xt = self.adain1[i].forward(x.clone(), s.clone());
            xt = Self::snake1d(xt, self.alpha1[i].val());
            xt = self.convs1[i].forward(xt);

            xt = self.adain2[i].forward(xt, s.clone());
            xt = Self::snake1d(xt, self.alpha2[i].val());
            xt = self.convs2[i].forward(xt);

            x = x + xt;
        }
        x
    }
}

/// Exact GPU/CPU STFT & iSTFT Engine for `Kokoro-82M` (`n_fft = 20, hop = 5, win = 20`).
/// Uses precomputed Hann-windowed Fourier basis kernels (`[22, 1, 20]`) and COLA squared-window normalization.
pub struct KokoroStftBasis {
    pub forward_basis: Param<Tensor<3>>,
    pub inverse_basis: Param<Tensor<3>>,
    pub window_sq_kernel: Param<Tensor<3>>,
}

impl KokoroStftBasis {
    pub fn new(device: &Device) -> Self {
        let n = POST_N_FFT;
        let bins = FREQ_BINS; // 11
        let mut hann = vec![0.0f32; n];
        for (t, h_slot) in hann.iter_mut().enumerate() {
            *h_slot = 0.5 * (1.0 - (2.0 * PI * (t as f32) / (n as f32)).cos());
        }

        let mut fwd_weights = vec![0.0f32; bins * 2 * n];
        let mut inv_weights = vec![0.0f32; bins * 2 * n];

        for k in 0..bins {
            let inv_scale = if k == 0 || k == n / 2 {
                1.0 / (n as f32)
            } else {
                2.0 / (n as f32)
            };

            for t in 0..n {
                let angle = 2.0 * PI * (k as f32) * (t as f32) / (n as f32);
                let cos_val = angle.cos();
                let sin_val = angle.sin();

                fwd_weights[k * n + t] = hann[t] * cos_val;
                fwd_weights[(bins + k) * n + t] = -hann[t] * sin_val;

                inv_weights[k * n + t] = hann[t] * cos_val * inv_scale;
                inv_weights[(bins + k) * n + t] = -hann[t] * sin_val * inv_scale;
            }
        }

        let win_sq: Vec<f32> = hann.iter().map(|&w| w * w).collect();

        Self {
            forward_basis: Param::from_tensor(
                Tensor::<1>::from_floats(fwd_weights.as_slice(), device).reshape([bins * 2, 1, n]),
            )
            .set_require_grad(false),
            inverse_basis: Param::from_tensor(
                Tensor::<1>::from_floats(inv_weights.as_slice(), device).reshape([bins * 2, 1, n]),
            )
            .set_require_grad(false),
            window_sq_kernel: Param::from_tensor(
                Tensor::<1>::from_floats(win_sq.as_slice(), device).reshape([1, 1, n]),
            )
            .set_require_grad(false),
        }
    }

    /// Forward STFT on 1D audio `[B, 1, Samples]` -> `(Magnitude [B, 11, Frames], Phase [B, 11, Frames])`.
    pub fn transform(&self, audio: Tensor<3>) -> (Tensor<3>, Tensor<3>) {
        let pad = POST_N_FFT / 2;
        let spec_re_im = conv1d(
            audio,
            self.forward_basis.val(),
            None,
            ConvOptions::new([POST_HOP_SIZE], [pad], [1], 1),
        );

        let parts = spec_re_im.chunk(2, 1);
        let (re, im) = (parts[0].clone(), parts[1].clone());

        let mag = (re.clone().powf_scalar(2.0) + im.clone().powf_scalar(2.0) + 1e-9).sqrt();
        let phase = Self::tensor_atan2(im, re);
        (mag, phase)
    }

    /// Inverse STFT (`Magnitude [B, 11, F]`, `Phase [B, 11, F]`) -> `Audio [B, 1, T_samples]` (`24 kHz`).
    pub fn inverse(&self, magnitude: Tensor<3>, phase: Tensor<3>) -> Tensor<3> {
        let re = magnitude.clone() * phase.clone().cos();
        let im = magnitude * phase.sin();

        let re_im = Tensor::cat(vec![re, im], 1);
        let [b, _, frames] = re_im.dims();
        let pad = POST_N_FFT / 2;

        let raw_ola = conv_transpose1d(
            re_im,
            self.inverse_basis.val(),
            None,
            ConvTransposeOptions::new([POST_HOP_SIZE], [pad], [0], [1], 1),
        );

        let ones = Tensor::<3>::ones([b, 1, frames], &raw_ola.device());
        let win_envelope = conv_transpose1d(
            ones,
            self.window_sq_kernel.val(),
            None,
            ConvTransposeOptions::new([POST_HOP_SIZE], [pad], [0], [1], 1),
        );

        raw_ola / win_envelope.clamp_min(1e-8)
    }

    /// Differentiable minimax polynomial `atan2(y, x)` on Burn tensors.
    pub fn tensor_atan2(y: Tensor<3>, x: Tensor<3>) -> Tensor<3> {
        let r = (x.clone().powf_scalar(2.0) + y.clone().powf_scalar(2.0) + 1e-12).sqrt();
        let u = y / (r + x + 1e-12);
        let u2 = u.clone().powf_scalar(2.0);
        let atan_u = u * (u2.clone() * (u2 * 0.07765095 - 0.28743447) + 0.995354);
        atan_u * 2.0
    }
}

/// Complete `KokoroIstftGenerator` (`decoder.generator` in `kokoro-v1_0.safetensors`).
/// Converts `[Batch, in_channels, Frames]`, speaker style `[Batch, style_dim]`, and `F0` `[Batch, 1, Frames*2]`
/// into a `24 kHz` waveform `[Batch, 1, Frames * 300]`.
pub struct KokoroIstftGenerator {
    pub m_source_l_linear: Linear,
    pub ups: Vec<ConvTranspose1d>,
    pub noise_convs: Vec<Conv1d>,
    pub noise_res: Vec<AdaInResBlock1>,
    pub resblocks: Vec<AdaInResBlock1>,
    pub conv_post: Conv1d,
    pub stft_engine: KokoroStftBasis,
    pub num_kernels: usize,
    pub num_upsamples: usize,
}

impl KokoroIstftGenerator {
    /// Initialize with configurable channel scale (`scale = 1` for full 512->256->128 `Kokoro-82M`,
    /// or `scale = 8` for fast 64->32->16 unit/edge execution).
    pub fn init_scaled(style_dim: usize, scale: usize, device: &Device) -> Self {
        let s = scale.max(1);
        let c0 = (512 / s).max(16);
        let c1 = (256 / s).max(8);
        let c2 = (128 / s).max(4);

        let ups = vec![
            ConvTranspose1dConfig::new([c0, c1], 20)
                .with_stride(10)
                .with_padding(5)
                .init(device),
            ConvTranspose1dConfig::new([c1, c2], 12)
                .with_stride(6)
                .with_padding(3)
                .init(device),
        ];

        let noise_convs = vec![
            Conv1dConfig::new(22, c1, 12)
                .with_stride(6)
                .with_padding(PaddingConfig1d::Explicit(3, 3))
                .init(device),
            Conv1dConfig::new(22, c2, 1)
                .with_stride(1)
                .with_padding(PaddingConfig1d::Explicit(0, 0))
                .init(device),
        ];

        let noise_res = vec![
            AdaInResBlock1::init(c1, 7, [1, 3, 5], style_dim, device),
            AdaInResBlock1::init(c2, 11, [1, 3, 5], style_dim, device),
        ];

        let mut resblocks = Vec::with_capacity(6);
        for &ch in &[c1, c2] {
            for &k in &[3, 7, 11] {
                resblocks.push(AdaInResBlock1::init(ch, k, [1, 3, 5], style_dim, device));
            }
        }

        Self {
            m_source_l_linear: LinearConfig::new(HARMONIC_CHANNELS, 1).init(device),
            ups,
            noise_convs,
            noise_res,
            resblocks,
            conv_post: Conv1dConfig::new(c2, FREQ_BINS * 2, 7)
                .with_padding(PaddingConfig1d::Explicit(3, 3))
                .init(device),
            stft_engine: KokoroStftBasis::new(device),
            num_kernels: 3,
            num_upsamples: 2,
        }
    }

    pub fn init(style_dim: usize, device: &Device) -> Self {
        Self::init_scaled(style_dim, 1, device)
    }

    /// Generate 9-channel phase-continuous harmonic sine excitation (`SourceModuleHnNSF`) from `F0`.
    pub fn synthesize_harmonic_source(&self, f0: Tensor<3>, total_samples: usize) -> Tensor<3> {
        let [b, _, f0_frames] = f0.dims();
        let device = f0.device();
        let scale_factor = (total_samples / f0_frames.max(1)).max(1);

        let f0_upsampled = f0
            .swap_dims(1, 2)
            .repeat_dim(2, scale_factor)
            .reshape([b, f0_frames * scale_factor, 1]);

        let multipliers = Tensor::<1>::from_floats(
            [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            &device,
        )
        .reshape([1, 1, HARMONIC_CHANNELS]);

        let freq_matrix = f0_upsampled.clone() * multipliers;
        let rad_per_sample = (freq_matrix / (KOKORO_SAMPLE_RATE as f32)) * (2.0 * PI);
        let phase_accum = Self::causal_cumsum_dim1(rad_per_sample);

        let voiced_mask = f0_upsampled.greater_elem(10.0).float();
        let sine_waves = phase_accum.sin() * 0.1 * voiced_mask;

        let har_source = activation::tanh(self.m_source_l_linear.forward(sine_waves));
        har_source.swap_dims(1, 2)
    }

    /// Parallel prefix sum (Hillis-Steele $O(\log T)$ scan) along dim 1 (`[B, T, C]`).
    pub fn causal_cumsum_dim1(mut x: Tensor<3>) -> Tensor<3> {
        let [b, t, c] = x.dims();
        let mut stride = 1;
        while stride < t {
            let left = x.clone().slice([0..b, 0..(t - stride), 0..c]);
            let right = x.clone().slice([0..b, stride..t, 0..c]);
            let head = x.slice([0..b, 0..stride, 0..c]);
            x = Tensor::cat(vec![head, right + left], 1);
            stride *= 2;
        }
        x
    }

    /// Complete End-to-End `KokoroIstftGenerator` forward pass (`[B, C0, Frames]` -> `[B, 1, Frames * 300]` @ `24 kHz`).
    pub fn forward(&self, mut x: Tensor<3>, s: Tensor<2>, f0: Tensor<3>) -> Tensor<3> {
        let [_, _, in_frames] = x.dims();
        let total_samples = in_frames * 300;

        let har_waveform = self.synthesize_harmonic_source(f0, total_samples);
        let (har_spec, har_phase) = self.stft_engine.transform(har_waveform);
        let har = Tensor::cat(vec![har_spec, har_phase], 1);

        for i in 0..self.num_upsamples {
            x = activation::leaky_relu(x, 0.1);
            x = self.ups[i].forward(x);

            let mut x_source = self.noise_convs[i].forward(har.clone());
            let target_t = x.dims()[2];
            let src_t = x_source.dims()[2];
            if src_t > target_t {
                let [b, c, _] = x_source.dims();
                x_source = x_source.slice([0..b, 0..c, 0..target_t]);
            }
            x_source = self.noise_res[i].forward(x_source, s.clone());
            x = x + x_source;

            let base_idx = i * self.num_kernels;
            let mut xs = self.resblocks[base_idx].forward(x.clone(), s.clone());
            for j in 1..self.num_kernels {
                xs = xs + self.resblocks[base_idx + j].forward(x.clone(), s.clone());
            }
            x = xs / (self.num_kernels as f32);
        }

        x = activation::leaky_relu(x, 0.01);
        let spec_phase_logits = self.conv_post.forward(x);

        let parts = spec_phase_logits.chunk(2, 1);
        let (raw_log_mag, raw_phase) = (parts[0].clone(), parts[1].clone());

        let spec_mag = raw_log_mag.clamp(-10.0, 10.0).exp();
        let spec_phase = raw_phase.sin();

        let audio_24k = self.stft_engine.inverse(spec_mag, spec_phase);
        audio_24k.clamp(-1.0, 1.0)
    }
}

/// Differentiable Multi-Resolution STFT Loss ($\mathcal{L}_{\text{MR-STFT}}$) combining spectral convergence
/// and log-magnitude $L_1$ distance for `Kokoro-82M` and Stage 4 GRPO waveform alignment.
pub fn multi_res_stft_loss(
    stft_basis: &KokoroStftBasis,
    pred_wav: Tensor<3>,
    target_wav: Tensor<3>,
) -> Tensor<1> {
    let (pred_mag, _) = stft_basis.transform(pred_wav);
    let (targ_mag, _) = stft_basis.transform(target_wav);

    let diff_norm = (targ_mag.clone() - pred_mag.clone())
        .powf_scalar(2.0)
        .sum()
        .sqrt();
    let targ_norm = targ_mag.clone().powf_scalar(2.0).sum().sqrt().clamp_min(1e-7);
    let sc_loss = diff_norm / targ_norm;

    let log_l1 = ((targ_mag + 1e-7).log() - (pred_mag + 1e-7).log())
        .abs()
        .mean();

    sc_loss + log_l1
}
