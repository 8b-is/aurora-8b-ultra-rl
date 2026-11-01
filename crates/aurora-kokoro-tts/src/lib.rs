pub mod kokoro_istft;

pub use kokoro_istft::{
    multi_res_stft_loss, AdaIn1d, AdaInResBlock1, KokoroIstftGenerator, KokoroStftBasis,
    FREQ_BINS, HARMONIC_CHANNELS, KOKORO_SAMPLE_RATE, POST_HOP_SIZE, POST_N_FFT, POST_WIN_SIZE,
};

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Tensor};

    #[test]
    fn test_kokoro_istft_generator_waveform_shape_and_finiteness() {
        let device = Device::default();
        let gen = KokoroIstftGenerator::init_scaled(32, 8, &device);

        // 4 input acoustic frames at 80 Hz -> 4 * 300 = 1200 samples (50 ms at 24 kHz)
        let x = Tensor::<3>::ones([1, 64, 4], &device) * 0.1;
        let style = Tensor::<2>::ones([1, 32], &device) * 0.2;
        let f0 = Tensor::<3>::full([1, 1, 8], 180.0, &device);

        let wav = gen.forward(x, style, f0);
        let [b, c, samples] = wav.dims();
        assert_eq!(b, 1);
        assert_eq!(c, 1);
        assert_eq!(samples, 1195);

        let data = wav.into_data().try_to_vec::<f32>().unwrap();
        assert!(data.iter().all(|v| v.is_finite() && *v >= -1.0 && *v <= 1.0));
    }

    #[test]
    fn test_multi_res_stft_loss_zero_on_identical_waveforms() {
        let device = Device::default();
        let basis = KokoroStftBasis::new(&device);
        let wav = Tensor::<3>::ones([1, 1, 600], &device) * 0.25;
        let loss = multi_res_stft_loss(&basis, wav.clone(), wav);
        let val = loss.into_data().try_to_vec::<f32>().unwrap()[0];
        assert!(val < 1e-4, "identical waveforms must yield ~0 MR-STFT loss, got {val}");
    }
}
