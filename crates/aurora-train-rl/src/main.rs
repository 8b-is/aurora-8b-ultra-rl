use std::path::PathBuf;

use anyhow::Result;
use aurora_train_rl::AuroraE2eTrainer;
use aurora_weights::Lfm2MoeConfig;
use burn::tensor::Device;
use tracing::info;

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let stage = std::env::args()
        .skip_while(|a| a != "--stage")
        .nth(1)
        .unwrap_or_else(|| "all".to_string());

    let steps: usize = std::env::args()
        .skip_while(|a| a != "--steps")
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);

    let out_dir = std::env::args()
        .skip_while(|a| a != "--out-dir")
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("./checkpoints"));

    let device = Device::default();
    let cfg = Lfm2MoeConfig::compact_test();
    let trainer = AuroraE2eTrainer::init(cfg, device);

    info!(
        stage = %stage,
        steps,
        out_dir = %out_dir.display(),
        "Starting Aurora-8B-Ultra-RL 5-Stage Pure-Rust Burn Training Pipeline"
    );

    let reports = trainer.run_pipeline(&stage, steps, &out_dir)?;
    println!("{}", serde_json::to_string_pretty(&reports)?);

    Ok(())
}
