# Aurora-8B-Ultra-RL (`aurora-8b-ultra-rl`)

**100% Pure-Rust End-to-End $p/q$ Quantal Full-Duplex Audio-Language Model** built natively on **Burn (`0.22.1`)** and **CubeCL (`0.5`+)** for **ESP32-S3 / ESP32-P4** bare-metal edge nodes and **Discord Full-Duplex Voice Bots**.

---

## 1. Unified Model Synergy (`Aurora-8B-Liquid-Kokoro`)

`Aurora-8B-Ultra-RL` fuses four state-of-the-art open-weights architectures into a single pure-Rust Burn computation graph:

| Subsystem | Source Checkpoints | Architectural Role in `Aurora-8B-Ultra-RL` |
| :--- | :--- | :--- |
| **8B Hybrid MoE Backbone** | `LiquidAI/LFM2.5-8B-A1B` | **8.3B total / 1.5B active parameters** across 24 layers: **18 Double-Gated ShortConv layers** (`conv_L_cache = 3`) + **6 GQA layers** (`32` Q / `8` KV heads, `QK-RMSNorm`, $\theta = 10^6$) at `[2, 6, 10, 14, 18, 21]`, with 2 dense SwiGLU layers (`0..2`) and 22 **Sparse 32-Expert MoE layers** (`top-4`, normalized sigmoid affinity routing). |
| **RL Teacher & Speculative MTP** | `XiaomiMiMo/MiMo-V2.6-Pro-RL` | Provides the **5-layer Multi-Token Prediction (MTP)** head drafting 5 future tokens per step, plus **MOPD2** (Multi-Teacher On-Policy Distillation) and **GRPO** (Group-Relative Policy Optimization). |
| **Streaming ASR, 80ms Clock & 8-Class VAD** | `Edge0/Audio8-ASR-Infinite` + `LiquidAI/LFM2.5-Audio-1.5B` | **8-Codebook RVQ Depthformer** (`q=8`, `dim=256`, `layers=6`, codes `0..2047`, stop `2048`) and **8-Class Multi-Horizon Semantic VAD** (`Silence`, `SpeechActive`, `ThinkingPause`, `EndOfTurn_0_25s`, `EndOfTurn_0_5s`, `EndOfTurn_1_0s`, `Backchannel`, `BargeInInterrupt`). |
| **Expressive TTS Synthesis** | `hexgrad/Kokoro-82M` + `Edge0/Audio8-TTS-Realistic-0.6b` | Native Burn `Kokoro-82M` (`StyleTTS2` + `ISTFTNet` 24 kHz generator with `AdaIN1d`, `Snake1d`, 9-harmonic `SourceModuleHnNSF` with Hillis-Steele $O(\log T)$ parallel scan, and exact COLA `conv_transpose1d` synthesis). |
| **Bare-Metal ESP32-S3 Edge Node** | `Edge0/Audio8-ASR-0.1B` (Distilled) | `#![no_std]` `burn-flex` INT8 depthwise-separable micro-encoder + 2-codebook RVQ quantizer + NLMS Acoustic Echo Cancellation (AEC) emitting 42-byte `Quantal80msPacket` frames (`4.2 kbps`). |

---

## 2. The $p/q$ Quantal Objective

Continuous speech is discretized at an `80 ms` (`12.5 Hz`) frame clock into $p=1$ **semantic audio token** and $q=8$ **acoustic RVQ codebooks** alongside an 8-class **Semantic VAD distribution**:

$$
\mathcal{L}_{\text{Aurora}} = \sum_{t=1}^{T} \left[ \alpha \log P_{\theta}(s_t \mid \mathbf{h}_t) + \beta \sum_{j=1}^{8} \log P_{\theta}(a_{t,j} \mid \mathbf{h}_t, s_t, a_{t,<j}) + \gamma \mathcal{L}_{\text{VAD}}(\mathbf{v}_t \mid \mathbf{h}_t) + \lambda_{\text{STFT}} \mathcal{L}_{\text{MR-STFT}}(\hat{y}, y) \right]
$$

---

## 3. Repository Structure

```text
aurora-8b-ultra-rl/
├── Cargo.toml                      # Workspace root (9 crates)
├── Justfile                        # Build, test, weight export, RL train, ESP32 flash & multi-remote push
├── bootstrap_aurora.sh             # Multi-remote GitHub / Codeberg / g8t & OpenCode DeepSeek-V4 bootstrap
├── opencode.json                   # OpenCode configuration for deepseek/deepseek-v4
├── PROMPT_ULTRAOMNI.md             # Master One-Shot UltraOmni architectural specification
└── crates/
    ├── aurora-core/                # #[no_std] Quantal80msPacket (42B), NLMS AEC, polyphase FIR resamplers
    ├── aurora-weights/             # Safetensors -> Burnpack (.bpk) converter, LFM2.5-8B-A1B & Kokoro-82M modules
    ├── aurora-kernels/             # Fused LFM2.5 ShortConv, 8-codebook Depthformer & Autodiff backward bridge
    ├── aurora-kokoro-tts/          # Pure-Rust Burn Kokoro-82M ISTFTNet (N_FFT=20, hop=5, win=20) + MR-STFT loss
    ├── aurora-train-rl/            # 5-stage RL pipeline (Projector -> SFT -> MOPD2 -> GRPO -> Quant) + 5-layer MTP
    ├── aurora-discord/             # Serenity + Songbird full-duplex RTP voice engine + <1ms barge-in flush
    ├── aurora-esp32-gateway/       # Axum /v1/aurora-esp32 WebSocket server + 960B (30ms) RFC-6455 chunk pacer
    ├── aurora-unified-daemon/      # Single-process GPU daemon sharing 5.2GB VRAM across Discord & ESP32 streams
    └── aurora-esp32-node/          # Bare-metal ESP32-S3 (#![no_std] esp-hal + burn-flex) firmware & WS parser
```

---

## 4. Quick Start

```bash
# 1. Type-check and run all unit & architectural invariant tests across the workspace
cargo test --workspace

# 2. Run the 5-stage RL training pipeline (Projector -> SFT -> MOPD2 -> GRPO -> Quant)
cargo run --release -p aurora-train-rl -- --stage all

# 3. Launch the unified full-duplex GPU daemon (Discord Voice + Axum ESP32 WebSocket Gateway)
DISCORD_TOKEN="your_bot_token" \
ESP32_WS_BIND="0.0.0.0:8080" \
cargo run --release -p aurora-unified-daemon
```
