use anyhow::Result;
use aurora_esp32_gateway::{build_esp32_router, Esp32GatewayEngine};
use burn::tensor::Device;
use tracing::info;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let bind_addr =
        std::env::var("ESP32_WS_BIND").unwrap_or_else(|_| "0.0.0.0:8080".to_string());
    let device = Device::default();
    let engine = Esp32GatewayEngine::new(device);
    let _app = build_esp32_router(engine);

    info!(
        bind = %bind_addr,
        "Aurora-8B-Ultra-RL Axum /v1/aurora-esp32 WebSocket Gateway initialized."
    );
    Ok(())
}
