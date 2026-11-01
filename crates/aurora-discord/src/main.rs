use std::sync::Arc;

use anyhow::Result;
use aurora_discord::{AuroraDiscordEngine, DuplexAudioSink};
use burn::tensor::Device;
use tokio::sync::mpsc;
use tracing::info;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let device = Device::default();
    let sink = DuplexAudioSink::new();
    let (tx, _rx) = mpsc::unbounded_channel();
    let _engine = Arc::new(AuroraDiscordEngine::new(device, sink, tx));

    info!(
        "Aurora-8B-Ultra-RL Discord Full-Duplex Voice Engine initialized (48kHz Stereo RTP <-> 80ms Quantal Clock)."
    );
    Ok(())
}
