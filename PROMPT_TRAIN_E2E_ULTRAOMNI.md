# OMNI ONE-SHOT PROMPT: END-TO-END PURE-RUST BURN + CUBECL TRAINING OF `AURORA-8B-ULTRA-RL`

> **Usage (OpenCode / DeepSeek-V4 / Claude / Gemini / Autonomous Coding Agent):**
> ```bash
> opencode run -m deepseek/deepseek-v4 "$(cat PROMPT_TRAIN_E2E_ULTRAOMNI.md)"
> ```

---

## ROLE & MISSION
You are **UltraOmni-Trainer**, an autonomous Staff ML Systems & Rust Compiler Engineer.
Your mission is to execute the complete, zero-Python, 100% pure-Rust **5-Stage End-to-End Training, Multi-Teacher On-Policy Distillation (`MOPD2`), Group-Relative Policy Optimization (`GRPO`), CubeCL INT4/FP8 Quantization, and ESP32-S3 Edge Micro-Distillation Pipeline** for **`Aurora-8B-Ultra-RL`** inside the `aurora-8b-ultra-rl` workspace using **Burn (`0.22.1`)** and **CubeCL (`0.5`+)**.

---

## 1. SOURCE CHECKPOINTS & ARCHITECTURAL FUSION MATRIX

Fuse the following five open-weights checkpoints into a single unified **Burn (`tracel-ai/burn`)** model graph (`Aurora-8B-Ultra-RL`):

| Subsystem | Source Hugging Face Checkpoint | Extracted Modules & Mathematical Role |
| :--- | :--- | :--- |
| **8B Hybrid MoE Trunk (`8.3B Total / 1.5B Active`)** | `LiquidAI/LFM2.5-8B-A1B` | **24 layers (`hidden_size = 2048`):** `18` Double-Gated Short-Range Causal Convolution blocks (`Lfm2MoeShortConv`, `conv_L_cache = 3`) + `6` Grouped-Query Attention blocks (`Lfm2MoeAttention`, `32` Q / `8` KV heads, `head_dim = 64`, `QK-RMSNorm`, $\theta_{\text{RoPE}} = 10^6$) at indices `[2, 6, 10, 14, 18, 21]`. Layers `0..2` use dense SwiGLU (`intermediate_size = 7168`); Layers `2..24` use **Sparse 32-Expert MoE** (`top_k = 4`, `moe_intermediate_size = 1792`, normalized sigmoid affinity routing + `expert_bias`). |
| **RL Teacher & 5-Layer Speculative MTP** | `XiaomiMiMo/MiMo-V2.6-Pro-RL` | Contributes the **308M AudioTokenizer (`8,192` semantic codes)**, the **5-layer Multi-Token Prediction (`MTP`) speculative head** (`MtpSpeculativeHead` drafting $s_{t+1..t+5}$ in one pass), and serves as the reasoning/tool-calling teacher in Stage 3 (`MOPD2`) and Stage 4 (`GRPO`). |
| **Streaming 80ms Clock, RVQ Depthformer & 8-Class VAD** | `Edge0/Audio8-ASR-Infinite` + `LiquidAI/LFM2.5-Audio-1.5B` | Contributes the **8-Codebook RVQ Audio Depthformer** (`DepthformerQ8Head`, `q = 8`, `codebook_vocab = 2049` with `0..2047` + stop `2048`) and the **8-Class Multi-Horizon Semantic VAD Head** (`[Silence, SpeechActive, ThinkingPause, EndOfTurn_0_25s, EndOfTurn_0_5s, EndOfTurn_1_0s, Backchannel, BargeInInterrupt]`) running at a locked `80 ms` (`12.5 Hz`) audio clock. |
| **Expressive 24kHz Acoustic Generator** | `hexgrad/Kokoro-82M` + `Edge0/Audio8-TTS-Realistic-0.6b` | Native Burn `KokoroIstftGenerator` (`StyleTTS2 + ISTFTNet`, `24 kHz`, $N_{\text{FFT}} = 20, \text{hop} = 5, \text{win} = 20$, `AdaIn1d`, `AdaInResBlock1` with `snake1d`, 9-harmonic `SourceModuleHnNSF` with Hillis-Steele $O(\log T)$ parallel prefix scan `causal_cumsum_dim1`, and exact COLA squared-window overlap-add via `conv_transpose1d`). |
| **Bare-Metal ESP32-S3 Edge Frontend (`380 KB`)** | `Edge0/Audio8-ASR-0.1B` (Distilled) | `#![no_std]` `burn-flex` INT8 depthwise-separable micro-encoder (`Esp32QuantalFrontend`) + 64-tap NLMS Acoustic Echo Canceller (`nlms_aec_cancel_frame`) emitting 42-byte `Quantal80msPacket` frames (`magic = 0xA8B0`, `4.2 kbps`) over RFC-6455 WebSockets. |

---

## 2. UNIFIED $p/q$ QUANTAL & FULL-DUPLEX TRAINING OBJECTIVE

At each `80 ms` (`12.5 Hz`) frame $t \in \{1..T\}$, `Aurora-8B-Ultra-RL` consumes an incoming acoustic frame (either `1,280` samples of `16 kHz` PCM from Discord RTP or a 42-byte `Quantal80msPacket` from an ESP32-S3 node) and optimizes the joint objective:

$$
\mathcal{L}_{\text{Aurora}}(\theta) = \sum_{t=1}^{T} \Big[
  \underbrace{\alpha \log P_{\theta}(s_t \mid \mathbf{h}_t)}_{\text{Slow AR Semantic/Text } (p=1)}
  + \underbrace{\beta \sum_{k=1}^{8} \log P_{\theta}(a_{t,k} \mid \mathbf{h}_t, s_t, a_{t,<k})}_{\text{Fast AR 8-Codebook RVQ } (q=8)}
  + \underbrace{\gamma \mathcal{L}_{\text{CE}}(\mathbf{v}_t, \mathbf{v}_t^* \mid \mathbf{h}_t)}_{\text{8-Class Semantic VAD}}
  + \underbrace{\lambda_{\text{MTP}} \sum_{m=1}^{5} 0.8^m \log P_{\theta}(s_{t+m} \mid \mathbf{h}_t^{(m)})}_{\text{5-Layer MiMo Speculative MTP}}
\Big] + \lambda_{\text{STFT}} \mathcal{L}_{\text{MR-STFT}}(\hat{y}_{24\text{k}}, y_{24\text{k}}^*)
$$

with loss weights $\alpha = 1.0$, $\beta = 0.5$, $\gamma = 0.4$, $\lambda_{\text{MTP}} = 0.3$, and $\lambda_{\text{STFT}} = 0.25$.

---

## 3. DATASET MIXING & AUGMENTATION SCHEDULE

| Dataset | Scale | Modality & Augmentation Role | Stage Mix Ratio |
| :--- | :--- | :--- | :--- |
| **Granary / YODAS2 / LibriHeavy** | `500,000 hrs` | Multilingual streaming ASR alignment & `80 ms` causal audio patch grounding. | Stage 1: `60%`, Stage 2: `25%` |
| **Emilia-YODAS / Emilia-101k** | `101,000 hrs` | Spontaneous conversational prosody, zero-shot voice cloning, and `Kokoro-82M` style conditioning. | Stage 1: `40%`, Stage 2: `35%` |
| **VoiceBench / UltraVoice / Seed-TTS-Eval** | `25,000 hrs` | Full-duplex overlapping speech, `<400ms` thinking pauses (`ThinkingPause`), and `<1ms` barge-in (`BargeInInterrupt`) trajectories. | Stage 2: `20%`, Stage 3/4: `50%` |
| **DNS-Challenge 5 + MUSAN + RIRs** | Noise + Room IRs | Simulating ESP32 `INMP441` MEMS mic self-noise, `MAX98357A` speaker echo leakage, Wi-Fi jitter, and Discord `48 kHz` Opus artifacts. | Applied online to `40%` of all audio frames |
| **MiMo-V2.6-RL-oss** | `1.2M trajectories` | Agentic tool-calling (`{"cmd":"gpio","pin":4,"val":1}` and Discord slash/voice commands) preserving 8B reasoning capacity. | Stage 2: `20%`, Stage 3/4: `50%` |

---

## 4. THE 5-STAGE END-TO-END BURN TRAINING PIPELINE

### Stage 0: Zero-Python Safetensors Import & `WeightNormConv1d` Fusion (`crates/aurora-weights`)
1. Parse `LiquidAI/LFM2.5-8B-A1B` (`config.json` + `.safetensors`) using `LFM2_MOE_KEY_REMAP_RULES`.
2. Expand `embed_tokens` and `lm_head` from `65,536` to `73,744` (`+8,192` semantic audio tokens + `16` duplex control/VAD tokens: `<|vad_silence|>`, `<|vad_active|>`, `<|vad_thinking|>`, `<|vad_eot_250ms|>`, `<|vad_eot_500ms|>`, `<|vad_eot_1000ms|>`, `<|vad_backchannel|>`, `<|vad_barge_in|>`, `<|esp32_gpio|>`, `<|voice_clone_ref|>`).
3. Parse `hexgrad/Kokoro-82M` (`kokoro-v1_0.safetensors`) using `KOKORO_82M_KEY_REMAP_RULES` and execute one-time `WeightNormConv1d::into_fused_conv1d()` ($W = g \cdot v / \|v\|_2$) across all `AdaInResBlock1` and `noise_convs` layers.
4. Export the unified initial checkpoint to `checkpoints/aurora_8b_stage0_init.bpk`.

### Stage 1: Multimodal Projector & Quantal $p/q$ Vocabulary Warmup (`15B` Tokens)
- **Frozen Parameters:** All 18 `Lfm2MoeShortConv` layers, 6 `Lfm2MoeAttention` layers, and 22 `Lfm2MoeSparseBlock` 32-expert MoE blocks.
- **Trainable Parameters:** Extended audio rows of `embed_tokens` (`65536..73744`), `Esp32QuantalAdapter` (`q0_embed`, `q1_embed`), `DepthformerQ8Head` codebook embeddings, and `kokoro_prosody_proj` (`2048 -> 512`).
- **Optimizer:** Burn `AdamW` ($\text{lr} = 2.0 \times 10^{-4}$, $\beta_1 = 0.9, \beta_2 = 0.95$, weight decay $0.01$, cosine decay to $2.0 \times 10^{-5}$, 2,000 warmup steps).

### Stage 2: Interleaved $p/q$ Full-Duplex SFT + 5-Layer `MTP` + `Kokoro` ISTFTNet (`40B` Tokens)
- **Unfrozen Parameters:** Full `LFM2.5-8B-A1B` (`1.5B` active params/token), `MtpSpeculativeHead` (5 draft blocks), `DepthformerQ8Head` (8 RVQ codebooks), `vad_head` (8 classes), and `KokoroIstftGenerator`.
- **Custom Autodiff Kernel:** Uses `aurora_kernels::fused_lfm_short_conv_step` on forward and analytical `aurora_kernels::fused_lfm_short_conv_backward` on backward to keep `18`-layer causal convolution states in SRAM.
- **Optimizer:** Burn `AdamW` ($\text{lr} = 3.0 \times 10^{-5}$ for trunk, $1.0 \times 10^{-4}$ for `MtpSpeculativeHead` and `DepthformerQ8Head`).

### Stage 3: `MOPD2` Multi-Teacher On-Policy Distillation (`10B` Tokens)
- Student `Aurora-8B-Ultra-RL` generates on-policy rollouts $y \sim \pi_{\theta}(\cdot \mid x)$ across interleaved speech/text/tool turns.
- Minimizes token-level and frame-level Reverse-KL divergence against the teacher ensemble:
  - **Reasoning & Tool Teacher:** `XiaomiMiMo/MiMo-V2.6-Pro-RL`
  - **80ms Streaming ASR & 8-Class Semantic VAD Teacher:** `Edge0/Audio8-ASR-Infinite`
  $$\mathcal{L}_{\text{MOPD2}}(\theta) = \mathbb{E}_{x, y \sim \pi_{\theta}} \left[ D_{\text{KL}}\!\left(\pi_{\theta}(\cdot \mid y_{<t}) \,\|\, \pi_{\text{MiMo}}(\cdot \mid y_{<t})\right) + D_{\text{KL}}\!\left(P_{\theta}(\mathbf{v}_t \mid \mathbf{h}_t) \,\|\, P_{\text{Audio8}}(\mathbf{v}_t \mid \mathbf{h}_t)\right) \right]$$

### Stage 4: `GRPO` Full-Duplex Reinforcement Learning ($G = 8$ Rollouts/Prompt)
- For each noisy ESP32 / Discord duplex prompt $x$, sample $G = 8$ trajectories $\{y_1, \dots, y_8\} \sim \pi_{\theta_{\text{old}}}(\cdot \mid x)$.
- Score each rollout $i \in \{1..8\}$ with the deterministic 4-part rubric (`DuplexRolloutRubric`):
  $$R_i = 0.35 R_{\text{WER}}(y_i) + 0.30 R_{\text{VAD}}(y_i) + 0.20 R_{\text{UTMOS}}(y_i) + 0.15 R_{\text{ToolJSON}}(y_i)$$
  where $R_{\text{VAD}}$ rewards holding the turn during `< 400 ms` (`< 5` ticks) thinking pauses and flushing audio in `< 1 ms` on `BargeInInterrupt`.
- Compute group-normalized advantages $\hat{A}_i = \frac{R_i - \mu_R}{\sigma_R + 10^{-8}}$ (`compute_group_relative_advantages`) and update $\theta$ via `grpo_step_loss`:
  $$\mathcal{L}_{\text{GRPO}}(\theta) = -\frac{1}{G}\sum_{i=1}^{G} \min\!\left(r_i(\theta)\hat{A}_i,\, \text{clip}(r_i(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_i\right) + \beta_{\text{KL}} D_{\text{KL}}(\pi_{\theta} \,\|\, \pi_{\text{ref}}) + \lambda_{\text{STFT}} \mathcal{L}_{\text{MR-STFT}}(\hat{y}_i, y^*)$$

### Stage 5: INT4/FP8 Server Burnpack Export & ESP32-S3 `#![no_std]` Micro-Distillation
1. **Server Quantization (`5.2 GB` VRAM):** Quantize the 22 `Lfm2MoeSparseBlock` 32-expert weights to `INT4` (`AWQ` group-size `128`), keep the 18 `Lfm2MoeShortConv` kernels and `KokoroIstftGenerator` in `BF16/FP8`, and export `checkpoints/aurora_8b_ultra_rl.bpk` + `aurora_8b_ultra_rl_manifest.json`.
2. **ESP32-S3 Edge Distillation (`380 KB` PSRAM):** Distill the first 2 layers of the audio encoder + 2-codebook RVQ (`rvq_q0[10]`, `rvq_q1[8]`) + local 2-class VAD head into `crates/aurora-esp32-node`'s `Esp32QuantalFrontend` and export `checkpoints/esp32s3_quantal_micro.bin`.

---

## 5. AUTONOMOUS EXECUTION COMMANDS

Run the complete end-to-end verification and 5-stage training pipeline in pure Rust:

```bash
# 1. Verify all 9 workspace crates and 22 unit/integration tests
cargo test --workspace

# 2. Fuse Safetensors weights & verify WeightNormConv1d -> Conv1d export
cargo run --release -p aurora-weights

# 3. Execute all 5 training stages end-to-end (Projector -> SFT+MTP -> MOPD2 -> GRPO -> Quant+ESP32)
cargo run --release -p aurora-train-rl -- --stage all --steps 8 --out-dir ./checkpoints

# 4. Verify ESP32-S3 #[no_std] 80ms Quantal80msPacket (42B) encoding + NLMS AEC + RFC-6455 framing
cargo run --release -p aurora-esp32-node

# 5. Launch the unified single-process Discord + ESP32 full-duplex GPU inference daemon
cargo run --release -p aurora-unified-daemon
```
