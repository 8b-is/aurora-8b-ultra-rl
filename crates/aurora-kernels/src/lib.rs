pub mod autodiff_bridge;
pub mod depthformer_q8;
pub mod lfm_short_conv;

pub use autodiff_bridge::{fused_lfm_short_conv_backward, silu_backward};
pub use depthformer_q8::fused_depthformer_q8_step;
pub use lfm_short_conv::fused_lfm_short_conv_step;

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Tensor};

    #[test]
    fn test_fused_lfm_short_conv_step_and_rolling_cache() {
        let device = Device::default();
        let (b, t, d, k) = (2, 5, 8, 3);
        let proj_bcx = Tensor::<3>::ones([b, t, d * 3], &device) * 0.5;
        let weight = Tensor::<3>::ones([d, 1, k], &device) * 0.25;
        let cache = Tensor::<3>::zeros([b, d, k - 1], &device);

        let (y, next_cache) = fused_lfm_short_conv_step(proj_bcx, weight, cache);
        assert_eq!(y.dims(), [b, t, d]);
        assert_eq!(next_cache.dims(), [b, d, k - 1]);
    }

    #[test]
    fn test_fused_lfm_short_conv_backward_shapes_and_finite_grads() {
        let device = Device::default();
        let (b, t, d, k) = (1, 4, 8, 3);
        let proj_bcx = Tensor::<3>::ones([b, t, d * 3], &device) * 0.4;
        let weight = Tensor::<3>::ones([d, 1, k], &device) * 0.3;
        let cache = Tensor::<3>::zeros([b, d, k - 1], &device);
        let grad_y = Tensor::<3>::ones([b, t, d], &device);

        let (grad_bcx, grad_w) = fused_lfm_short_conv_backward(grad_y, proj_bcx, weight, cache);
        assert_eq!(grad_bcx.dims(), [b, t, d * 3]);
        assert_eq!(grad_w.dims(), [d, 1, k]);

        let max_grad: f32 = grad_bcx
            .abs()
            .max()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap()[0];
        assert!(max_grad > 0.0 && max_grad.is_finite());
    }
}
