use std::net::SocketAddr;

use anyhow::Result;
use aurora_discord::DuplexAudioSink;
use aurora_esp32_gateway::{build_esp32_router, Esp32GatewayEngine};
use aurora_unified_daemon::{CrossBridgeEvent, SharedAuroraRuntime};
use burn::tensor::Device;
use tracing::info;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let device = Device::default();
    let shared_runtime = SharedAuroraRuntime::new(device.clone());
    let discord_sink = DuplexAudioSink::new();
    let esp32_engine = Esp32GatewayEngine::new(device);

    // Cross-bridge listener mirroring ESP32 & Discord voice turns
    let mut bridge_rx = shared_runtime.bridge_tx.subscribe();
    let sink_clone = discord_sink.clone();
    tokio::spawn(async move {
        while let Ok(evt) = bridge_rx.recv().await {
            if let CrossBridgeEvent::VoiceTurnCompleted {
                source,
                reply_text,
                kokoro_24k_mono,
                ..
            } = evt
            {
                info!(?source, %reply_text, "Cross-bridge mirroring 24kHz Kokoro TTS audio");
                sink_clone.push_kokoro_24k_mono(&kokoro_24k_mono);
            }
        }
    });

    let bind_addr: SocketAddr = std::env::var("ESP32_WS_BIND")
        .unwrap_or_else(|_| "0.0.0.0:8080".to_string())
        .parse()?;
    let _router = build_esp32_router(esp32_engine);

    info!(
        %bind_addr,
        "Aurora-8B-Ultra-RL Unified Single-Process GPU Daemon ready (Shared 5.2GB VRAM across Discord + ESP32-S3)."
    );

    Ok(())
}
