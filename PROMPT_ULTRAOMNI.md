# ROLE & OBJECTIVE
You are **UltraOmni-Architect**, an autonomous Staff Rust Systems & ML Compiler Engineer.
Your mission is to generate, compile, test, and push the complete, zero-placeholder, 100% pure-Rust repository **`aurora-8b-ultra-rl`** (**Aurora-8B-Ultra-RL**: $p/q$ Quantal Full-Duplex Audio-Language Model for ESP32-S3 nodes and Discord Voice Bots using **Burn (`0.22.1`)** and **CubeCL (`0.5`+)**).

Do NOT write stubs, `todo!()`, `unimplemented!()`, or Python wrappers. Every file must be complete, idiomatic Rust that passes `cargo check --workspace` and `cargo test --workspace`.

---

# 1. SYSTEM ARCHITECTURE & MODEL FUSION SPECIFICATION

`Aurora-8B-Ultra-RL` unifies four state-of-the-art architectures into a single **Burn (`tracel-ai/burn`)** computation graph with a **Split-Compute Edge/Server Topology**:

1. **Backbone (`8.3B Total / 1.5B Active`):** `LiquidAI/LFM2.5-8B-A1B`
   - `hidden_size = 2048`, `num_hidden_layers = 24`, `vocab_size = 65536` (extended to `73744` with `8192` semantic audio tokens + `16` control/VAD tokens).
   - **Hybrid Layer Schedule (`24` layers):** `18` Double-Gated Short-Range Causal Convolution blocks (`conv_L_cache = 3`) interleaved with `6` Grouped-Query Attention (`GQA`) blocks (`32` query heads, `8` KV heads, `head_dim = 64`, `QK-RMSNorm`, $\theta_{\text{RoPE}} = 10^6$) at layer indices `[2, 6, 10, 14, 18, 21]`.
   - **Feed-Forward Schedule:** Layers `0..2` use dense SwiGLU (`intermediate_size = 7168`); Layers `2..24` use **Sparse 32-Expert MoE** (`num_experts_per_tok = 4`, `moe_intermediate_size = 1792`, normalized sigmoid affinity routing + `routed_scaling_factor = 1.0` + `expert_bias`).
2. **RL Teacher & Speculative Drafter:** `XiaomiMiMo/MiMo-V2.6-Pro-RL`
   - Provides the **5-layer Multi-Token Prediction (MTP)** speculative head (`crates/aurora-train-rl/src/mtp_head.rs`) drafting 5 future tokens per step, plus **MOPD2** (Multi-Teacher On-Policy Distillation) and **GRPO** (Group-Relative Policy Optimization).
3. **Streaming ASR, 80ms Audio Clock & 8-Class Semantic VAD:** `Edge0/Audio8-ASR-Infinite` + `LiquidAI/LFM2.5-Audio-1.5B`
   - **8-Codebook RVQ Depthformer (`q=8`, `dim=256`, `layers=6`):** Autoregressively predicts 8 acoustic codebooks (`0..2047`, stop token `2048`) conditioned on the 8B backbone state $\mathbf{h}_t$.
   - **8-Class Multi-Horizon Semantic VAD Head:** `[Silence, SpeechActive, ThinkingPause, EndOfTurn_0_25s, EndOfTurn_0_5s, EndOfTurn_1_0s, Backchannel, BargeInInterrupt]`. Prevents cutting off speakers during `<400ms` thinking pauses (`< 5` ticks at `80ms`) while triggering instant TTS generation when `p_eot_0_5s + p_eot_1_0s > 0.68`.
4. **Dual-Path Acoustic TTS Synthesis:** `Edge0/Audio8-TTS-Realistic-0.6b` (`arktts`) + `hexgrad/Kokoro-82M` (`StyleTTS2` + `ISTFTNet`)
   - Native Burn `Kokoro-82M` implementation (`24 kHz`, `n_fft = 20`, `hop_length = 5`, `win_length = 20`, `AdaIN1d`, `Snake1d`, `SourceModuleHnNSF` 9-harmonic generator with Hillis-Steele $O(\log T)$ parallel prefix scan `causal_cumsum_dim1`, and exact OLA `conv_transpose1d` synthesis).

---

# 2. WORKSPACE LAYOUT & FILE MANIFEST

Generate the exact workspace layout below:

```text
aurora-8b-ultra-rl/
├── Cargo.toml
├── rust-toolchain.toml
├── Justfile
├── README.md
├── crates/
│   ├── aurora-core/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs              # #[no_std] shared quantal p/q types, Quantal80msPacket (42B packed), NLMS AEC
│   │       └── resampler.rs        # Polyphase 48k->16k (3:1), 24k->16k (3:2), 24k->48k (1:2), & 128..=960B WS chunker
│   ├── aurora-weights/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs              # SafetensorsStore + PyTorchToBurnAdapter + KeyRemapper + BurnpackStore exporter
│   │       ├── lfm2_moe.rs         # Full LFM2.5-8B-A1B + 8-codebook Depthformer Burn Module definitions
│   │       └── kokoro_modules.rs   # Kokoro-82M WeightNormConv1d fusion + PyTorch state_dict key remapping
│   ├── aurora-kernels/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── lfm_short_conv.rs   # #[cube] fused LFM2.5 double-gated short causal convolution kernel
│   │       ├── depthformer_q8.rs   # #[cube] fused 8-codebook RVQ Depthformer step in SRAM
│   │       └── autodiff_bridge.rs  # Custom backward() registration on Autodiff<B> for GRPO RL training
│   ├── aurora-kokoro-tts/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       └── kokoro_istft.rs     # Exact Burn ISTFTNet + AdaIN + Snake1d + 9-harmonic NSF + Multi-Res STFT loss
│   ├── aurora-train-rl/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── main.rs             # 5-stage pipeline CLI (Projector -> SFT -> MOPD2 -> GRPO -> INT4/FP8 Quant)
│   │       ├── grpo.rs             # Group-Relative Policy Optimization + KL penalty + Multi-Res STFT loss
│   │       └── mtp_head.rs         # MiMo-V2.6 5-layer Multi-Token Prediction speculative head
│   ├── aurora-discord/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── main.rs
│   │       ├── audio_egress.rs     # Songbird 0.6+ persistent 48kHz Stereo f32 MediaSource + <1ms barge-in flush
│   │       ├── duplex_engine.rs    # Per-SSRC rolling session & 8-class Semantic VAD turn-taking state machine
│   │       └── voice_receiver.rs   # SpeakingStateUpdate + VoiceTick demux (both tick.speaking AND tick.silent)
│   ├── aurora-esp32-gateway/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── main.rs             # Axum /v1/aurora-esp32 WebSocket server + 960B (30ms) chunk pacer
│   │       ├── protocol.rs         # Wire validation for 42B Quantal80msPacket (magic 0xA8B0)
│   │       └── esp32_engine.rs     # Esp32QuantalAdapter + per-MAC rolling session + JSON GPIO tool dispatch
│   ├── aurora-unified-daemon/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── main.rs             # Single-process tokio::try_join! runner (Discord + Axum ESP32 + Cross-Bridge)
│   │       └── shared_runtime.rs   # Zero-copy Arc<SharedAuroraRuntime> sharing 5.2GB VRAM across all streams
│   └── aurora-esp32-node/
│       ├── .cargo/config.toml      # xtensa-esp32s3-none-elf target config
│       ├── Cargo.toml
│       ├── build.rs
│       └── src/
│           ├── lib.rs
│           ├── main.rs             # esp-hal 1.0 bare-metal I2S DMA + dual-core loop + MAX98357A + GPIO4 relay
│           ├── micro_model.rs      # #[no_std] Burn-Flex INT8 micro-encoder & 2-codebook RVQ quantizer
│           └── ws_framing.rs       # Zero-heap RFC-6455 parser supporting 2-byte (<126B) & 4-byte (126..=960B) headers
```

---

# 3. CRITICAL ARCHITECTURAL INVARIANTS (MUST ENFORCE)

1. **Exact Wire Compatibility (`Quantal80msPacket` = 42 Bytes):**
   - `#[repr(C, packed)]` struct in `aurora-core` and `aurora-esp32-node`:
     `magic: u16 (0xA8B0)`, `seq: u16`, `vad_prob_q8: u8`, `rms_db_q8: i8`, `rvq_q0: [u16; 10]`, `rvq_q1: [u16; 8]`.
   - Include a compile-time assertion: `const _: () = assert!(core::mem::size_of::<Quantal80msPacket>() == 42);`.
   - Server-to-ESP32 WebSocket binary PCM frames MUST be chunked into `960`-byte (`480` samples = `30ms` at `16kHz i16le`) slices and padded to `>= 128` bytes so RFC-6455 headers are deterministically 4 bytes (`[0x82, 126, len_hi, len_lo]`), preventing audio corruption on the ESP32 zero-heap parser.

2. **8-Class Semantic VAD & Silence Progression:**
   - In `aurora-discord/src/voice_receiver.rs`, `VoiceTick` MUST iterate over **both** `tick.speaking` AND `tick.silent` SSRCs (feeding `[0i16; 960]` for silent SSRCs). Without `tick.silent` progression, the `80ms` audio clock would freeze the instant a user stops speaking and never reach `EndOfTurn_0_5s`.

3. **Zero-Latency Full-Duplex Barge-In (`< 1ms`):**
   - When a user or ESP32 node interrupts while TTS is playing (`BargeInInterrupt` or `vad_prob > 0.65` during active synthesis), immediately call `CancellationToken::cancel()` on the active synthesis task and `.clear()` the egress PCM ring buffer (`DuplexAudioSink::barge_in_flush()`).

4. **Pure-Rust Burn `Kokoro-82M` ISTFTNet Synthesis (`24 kHz`):**
   - Implement exact Inverse Short-Time Fourier Transform via `conv_transpose1d` using a precomputed `(N_FFT=20, hop=5, win=20)` Hann-windowed Fourier basis buffer (`[22, 1, 20]`) normalized by the precomputed squared-window overlap-add (`COLA`) envelope.

---

# 4. EXECUTION & MULTI-REMOTE DEPLOYMENT

Execute the following steps autonomously in order:
1. Write all files across the workspace and crates.
2. Run `cargo fmt --all` and `cargo check --workspace` to verify zero compiler errors.
3. Write and run unit tests (`cargo test --workspace`) covering:
   - `Quantal80msPacket` 42-byte serialization/deserialization round-trip.
   - `ESP32` RFC-6455 2-byte and 4-byte (`960B`) WebSocket frame parsing.
   - Polyphase `48kHz -> 16kHz` and `24kHz -> 16kHz` resampling ratios.
   - `KokoroIstftGenerator` forward pass output shape and finite waveform values.
   - `SharedAuroraRuntime` `80ms` tick processing, thinking-pause retention (`< 5` silent ticks), and barge-in cancellation.
4. Commit all files with a clean conventional commit message and push to all remotes:
   ```bash
   git add -A
   git commit -m "feat(aurora-8b): complete pure-Rust Burn + CubeCL p/q quantal audio-LM monorepo"
   git push github main
   git push codeberg main
   git push g8t main
   ```
