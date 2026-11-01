fn main() {
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let esp32_feature = std::env::var("CARGO_FEATURE_ESP32S3").is_ok();

    if target_arch == "xtensa" || esp32_feature {
        println!("cargo:rustc-link-arg-bins=-Tlinkall.x");
        println!("cargo:rustc-link-arg-bins=-Trom_functions.x");
    }

    println!("cargo:rerun-if-env-changed=WIFI_SSID");
    println!("cargo:rerun-if-env-changed=WIFI_PASS");
    println!("cargo:rerun-if-env-changed=AURORA_WS_HOST");
    println!("cargo:rerun-if-env-changed=AURORA_WS_PORT");
}
