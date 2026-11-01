use aurora_kokoro_tts::{multi_res_stft_loss, KokoroStftBasis};
use burn::tensor::{activation, Device, Tensor};

/// Hyperparameters for Group-Relative Policy Optimization (`GRPO`) + `MOPD2` distillation.
#[derive(Debug, Clone, Copy)]
pub struct GrpoHyperParams {
    pub group_size: usize,
    pub clip_eps: f32,
    pub kl_beta: f32,
    pub stft_loss_weight: f32,
}

impl Default for GrpoHyperParams {
    fn default() -> Self {
        Self {
            group_size: 8,
            clip_eps: 0.20,
            kl_beta: 0.04,
            stft_loss_weight: 0.25,
        }
    }
}

/// Multi-objective composite reward rubric for a full-duplex `Aurora-8B-Ultra-RL` rollout $i \in \{1..G\}$.
#[derive(Debug, Clone, Copy)]
pub struct DuplexRolloutRubric {
    /// Round-trip Whisper/Audio8 ASR Word Error Rate reward: $1.0 - \text{WER}(\hat{y}_i, y^*)$.
    pub r_wer: f32,
    /// 8-class Semantic VAD turn-taking & `<1ms` barge-in latency reward (`[-1.0, 1.0]`).
    pub r_turn_vad: f32,
    /// Acoustic UTMOS + speaker cosine similarity reward from `Kokoro-82M` / `arktts` (`[0.0, 1.0]`).
    pub r_utmos_prosody: f32,
    /// Exact JSON schema & GPIO/Discord tool-call execution reward (`0.0` or `1.0`).
    pub r_tool_json: f32,
}

impl DuplexRolloutRubric {
    /// Weighted composite reward $R_i = 0.35 R_{\text{WER}} + 0.30 R_{\text{VAD}} + 0.20 R_{\text{UTMOS}} + 0.15 R_{\text{Tool}}$.
    pub fn total_reward(&self) -> f32 {
        0.35 * self.r_wer
            + 0.30 * self.r_turn_vad
            + 0.20 * self.r_utmos_prosody
            + 0.15 * self.r_tool_json
    }
}

/// Compute group-normalized advantages across $G$ rollouts:
/// $$\hat{A}_i = \frac{R_i - \text{mean}(R_1, \dots, R_G)}{\text{std}(R_1, \dots, R_G) + 10^{-8}}$$
pub fn compute_group_relative_advantages(rubrics: &[DuplexRolloutRubric]) -> Vec<f32> {
    if rubrics.is_empty() {
        return Vec::new();
    }
    let rewards: Vec<f32> = rubrics.iter().map(DuplexRolloutRubric::total_reward).collect();
    let n = rewards.len() as f32;
    let mean = rewards.iter().sum::<f32>() / n;
    let var = rewards
        .iter()
        .map(|&r| (r - mean) * (r - mean))
        .sum::<f32>()
        / n;
    let std = var.sqrt() + 1e-8;
    rewards.into_iter().map(|r| (r - mean) / std).collect()
}

/// Executes a single `GRPO` + `MOPD2` + `Multi-Res STFT` loss calculation in Burn:
/// - `student_logits`: `[G, T, V]`
/// - `old_logits`: `[G, T, V]`
/// - `teacher_logits`: `[G, T, V]` (`MiMo-V2.6-Pro-RL` + `Audio8-ASR-Infinite` teacher ensemble)
/// - `advantages`: `[G]`
/// - `pred_wav_24k`, `target_wav_24k`: `[G, 1, Samples]`
pub fn grpo_step_loss(
    student_logits: Tensor<3>,
    old_logits: Tensor<3>,
    teacher_logits: Tensor<3>,
    advantages: &[f32],
    stft_basis: &KokoroStftBasis,
    pred_wav_24k: Tensor<3>,
    target_wav_24k: Tensor<3>,
    cfg: &GrpoHyperParams,
) -> Tensor<1> {
    let [g, _t, _v] = student_logits.dims();
    let device: Device = student_logits.device();

    let log_pi = activation::log_softmax(student_logits.clone(), 2);
    let log_pi_old = activation::log_softmax(old_logits, 2);
    let pi_prob = activation::softmax(student_logits, 2);
    let log_teacher = activation::log_softmax(teacher_logits, 2);

    // Token-level probability ratio r_t(theta) = exp(log_pi - log_pi_old) averaged over vocabulary
    let ratio = (log_pi.clone() - log_pi_old).exp().mean_dim(2).reshape([g, _t]);
    let adv_tensor = Tensor::<1>::from_floats(advantages, &device).reshape([g, 1]);

    let surr1 = ratio.clone() * adv_tensor.clone();
    let surr2 = ratio.clamp(1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_tensor;
    // Surrogate objective: min(surr1, surr2) = 0.5 * (s1 + s2 - |s1 - s2|)
    let min_surr = (surr1.clone() + surr2.clone() - (surr1 - surr2).abs()) * 0.5;
    let policy_loss = -min_surr.mean();

    // Reverse-KL teacher penalty D_KL(pi_theta || pi_teacher)
    let kl_div = (pi_prob * (log_pi - log_teacher)).sum_dim(2).mean();

    // Multi-Resolution STFT waveform reconstruction loss on Kokoro-82M output
    let stft_loss = multi_res_stft_loss(stft_basis, pred_wav_24k, target_wav_24k);

    policy_loss + kl_div * cfg.kl_beta + stft_loss * cfg.stft_loss_weight
}
