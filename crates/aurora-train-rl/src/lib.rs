pub mod dataset;
pub mod grpo;
pub mod mtp_head;
pub mod pipeline;

pub use dataset::{augment_esp32_mems_and_rir, write_pcm16_wav_24k};
pub use grpo::{
    compute_group_relative_advantages, grpo_step_loss, DuplexRolloutRubric, GrpoHyperParams,
};
pub use mtp_head::{MtpDraftBlock, MtpSpeculativeHead};
pub use pipeline::{AuroraE2eTrainer, StageTrainingReport};

#[cfg(test)]
mod tests {
    use super::*;
    use aurora_kokoro_tts::KokoroStftBasis;
    use aurora_weights::Lfm2MoeConfig;
    use burn::tensor::{Device, Tensor};

    #[test]
    fn test_mtp_5_layer_speculative_head_drafts_5_tokens() {
        let device = Device::default();
        let mtp = MtpSpeculativeHead::init(32, 256, 5, &device);
        let h_t = Tensor::<3>::ones([1, 1, 32], &device) * 0.25;
        let drafted = mtp.draft_future_tokens(h_t);
        assert_eq!(drafted.len(), 5);
        for tok in drafted {
            assert_eq!(tok.dims(), [1, 1]);
        }
    }

    #[test]
    fn test_grpo_advantages_are_zero_mean_unit_variance() {
        let rubrics = vec![
            DuplexRolloutRubric {
                r_wer: 0.96,
                r_turn_vad: 0.92,
                r_utmos_prosody: 0.89,
                r_tool_json: 1.0,
            },
            DuplexRolloutRubric {
                r_wer: 0.72,
                r_turn_vad: 0.40,
                r_utmos_prosody: 0.65,
                r_tool_json: 0.0,
            },
            DuplexRolloutRubric {
                r_wer: 0.88,
                r_turn_vad: 0.85,
                r_utmos_prosody: 0.82,
                r_tool_json: 1.0,
            },
            DuplexRolloutRubric {
                r_wer: 0.91,
                r_turn_vad: 0.90,
                r_utmos_prosody: 0.87,
                r_tool_json: 1.0,
            },
        ];
        let adv = compute_group_relative_advantages(&rubrics);
        assert_eq!(adv.len(), 4);
        let mean: f32 = adv.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5);
        assert!(adv[0] > adv[1]);

        let device = Device::default();
        let basis = KokoroStftBasis::new(&device);
        let s_logits = Tensor::<3>::ones([4, 6, 64], &device) * 0.3;
        let o_logits = Tensor::<3>::ones([4, 6, 64], &device) * 0.25;
        let t_logits = Tensor::<3>::ones([4, 6, 64], &device) * 0.32;
        let pred_wav = Tensor::<3>::ones([4, 1, 300], &device) * 0.1;
        let targ_wav = Tensor::<3>::ones([4, 1, 300], &device) * 0.12;

        let loss = grpo_step_loss(
            s_logits,
            o_logits,
            t_logits,
            &adv,
            &basis,
            pred_wav,
            targ_wav,
            &GrpoHyperParams::default(),
        );
        let loss_val = loss.into_data().try_to_vec::<f32>().unwrap()[0];
        assert!(loss_val.is_finite());
    }

    #[test]
    fn test_e2e_5_stage_training_pipeline_decreases_loss() {
        let device = Device::default();
        let trainer = AuroraE2eTrainer::init(Lfm2MoeConfig::compact_test(), device);
        let tmp_dir = std::env::temp_dir().join("aurora_8b_e2e_test_ckpt");
        let reports = trainer
            .run_pipeline("all", 3, &tmp_dir)
            .expect("5-stage pipeline must succeed");
        assert_eq!(reports.len(), 5);
        for r in &reports[..4] {
            assert!(
                r.final_loss <= r.initial_loss,
                "Stage {} loss must decrease ({} -> {})",
                r.stage_name,
                r.initial_loss,
                r.final_loss
            );
        }
    }
}
