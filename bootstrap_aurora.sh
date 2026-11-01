#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# AURORA-8B-ULTRA-RL — Multi-Remote & OpenCode (DeepSeek-V4) Bootstrap
# ==============================================================================

REPO_NAME="aurora-8b-ultra-rl"
GH_ORG="8b-is"
CODEBERG_USER="peterlodri-sec"
G8T_HOST="${G8T_HOST:-git@g8t.io:peterlodri-sec}" # Override if self-hosted g8t URL differs

echo "[1/5] Initializing local Git repository & directory structure..."
mkdir -p "${REPO_NAME}" && cd "${REPO_NAME}"
git init -b main

mkdir -p \
  crates/aurora-core/src \
  crates/aurora-weights/src \
  crates/aurora-kernels/src \
  crates/aurora-kokoro-tts/src \
  crates/aurora-train-rl/src \
  crates/aurora-discord/src \
  crates/aurora-esp32-gateway/src \
  crates/aurora-unified-daemon/src \
  crates/aurora-esp32-node/.cargo \
  crates/aurora-esp32-node/src

echo "[2/5] Writing OpenCode configuration for DeepSeek-V4..."
cat << 'EOF' > opencode.json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "deepseek/deepseek-v4",
  "small_model": "deepseek/deepseek-v4",
  "provider": {
    "deepseek": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "DeepSeek V4",
      "options": {
        "baseURL": "https://api.deepseek.com/v1",
        "apiKey": "{env:DEEPSEEK_API_KEY}"
      },
      "models": {
        "deepseek-v4": {
          "name": "DeepSeek-V4 (1M Context / Reasoning + Code)",
          "limit": {
            "context": 1000000,
            "output": 65536
          }
        }
      }
    }
  }
}
EOF

echo "[3/5] Provisioning Remote Repositories..."

# 1. GitHub: 8b-is/aurora-8b-ultra-rl
if command -v gh &> /dev/null; then
  gh repo create "${GH_ORG}/${REPO_NAME}" --public \
    --description "Aurora-8B-Ultra-RL: 100% Pure-Rust Burn + CubeCL p/q Quantal Full-Duplex Audio-LM (LFM2.5-8B-A1B + Kokoro-82M + ESP32-S3 + Discord)" \
    || true
fi
git remote remove github 2>/dev/null || true
git remote add github "https://github.com/${GH_ORG}/${REPO_NAME}.git"

# 2. Codeberg: codeberg.org/peterlodri-sec/aurora-8b-ultra-rl
if [[ -n "${CODEBERG_TOKEN:-}" ]]; then
  curl -s -X POST "https://codeberg.org/api/v1/user/repos" \
    -H "Authorization: token ${CODEBERG_TOKEN}" \
    -H "Content-Type: application/json" \
    -d "{\"name\": \"${REPO_NAME}\", \"private\": false, \"default_branch\": \"main\", \"description\": \"Aurora-8B-Ultra-RL Pure-Rust Burn Audio-LM\"}" \
    > /dev/null || true
fi
git remote remove codeberg 2>/dev/null || true
git remote add codeberg "https://codeberg.org/${CODEBERG_USER}/${REPO_NAME}.git"

# 3. g8t: peterlodri-sec/aurora-8b-ultra-rl
git remote remove g8t 2>/dev/null || true
git remote add g8t "${G8T_HOST}/${REPO_NAME}.git"

echo "[4/5] Configuring unified 'origin' push-all remote..."
git remote remove origin 2>/dev/null || true
git remote add origin "https://github.com/${GH_ORG}/${REPO_NAME}.git"
git remote set-url --add --push origin "https://github.com/${GH_ORG}/${REPO_NAME}.git"
git remote set-url --add --push origin "https://codeberg.org/${CODEBERG_USER}/${REPO_NAME}.git"
git remote set-url --add --push origin "${G8T_HOST}/${REPO_NAME}.git"

echo "[5/5] Ready! Run: opencode run -m deepseek/deepseek-v4 \"\$(cat PROMPT_ULTRAOMNI.md)\""
