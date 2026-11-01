# Aurora-8B-Ultra-RL — Pure-Rust Burn + CubeCL Full-Duplex Audio-LM Monorepo

default: check test

# Type-check all workspace crates
check:
    cargo check --workspace

# Run full test suite across all workspace crates
test:
    cargo test --workspace

# Build release binaries for unified daemon, trainer, Discord bot, and ESP32 gateway
build-release:
    cargo build --release --workspace

# Export HuggingFace Safetensors (LFM2.5-8B-A1B + Kokoro-82M) to mmap Burnpack (.bpk)
convert-weights:
    cargo run --release -p aurora-weights -- \
        --lfm-dir ./checkpoints/LFM2.5-8B-A1B \
        --kokoro-dir ./checkpoints/Kokoro-82M \
        --out-bpk ./checkpoints/aurora_8b_ultra_rl.bpk

# Run 5-stage RL training (Projector -> SFT -> MOPD2 -> GRPO -> Quant)
train stage="all":
    cargo run --release -p aurora-train-rl -- --stage {{stage}}

# Run the unified single-process Discord + ESP32 full-duplex GPU daemon
run-daemon:
    cargo run --release -p aurora-unified-daemon

# Flash ESP32-S3 bare-metal node over USB-Serial/JTAG
flash-esp32:
    cd crates/aurora-esp32-node && cargo +esp build --release --features esp32s3 && espflash flash --monitor target/xtensa-esp32s3-none-elf/release/aurora-esp32-node

# Push main branch to all configured version control remotes (GitHub, Codeberg, g8t)
push-all:
    LEFTHOOK=0 git push github main
    LEFTHOOK=0 git push peterlodri-sec main || true
    LEFTHOOK=0 git push g8t-github main || true
    LEFTHOOK=0 git push codeberg main || true
    LEFTHOOK=0 git push g8t main || true
