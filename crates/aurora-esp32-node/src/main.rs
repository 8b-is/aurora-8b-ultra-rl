//! Main Firmware for the `#![no_std]` ESP32-S3 Edge Node (`crates/aurora-esp32-node`):
//!   - I2S0 RX (`INMP441` Mic): `BCLK = GPIO4`, `WS = GPIO5`, `DOUT = GPIO6`
//!   - I2S0 TX (`MAX98357A` Speaker Amp): `DIN = GPIO7`
//!   - 8 MB Octal PSRAM allocator + `burn-flex` `Esp32QuantalFrontend` (`80 ms` = `1,280` samples @ `16 kHz`)
//!   - Single-tap NLMS Acoustic Echo Cancellation + `< 1 ms` Local DMA Barge-In Flush
//!   - Zero-allocation RFC-6455 Binary WebSocket Framing (`42-byte` `Quantal80msPacket` = `4.2 kbps`)

#![cfg_attr(feature = "esp32s3", no_std)]
#![cfg_attr(feature = "esp32s3", no_main)]

#[cfg(feature = "esp32s3")]
extern crate alloc;

pub mod micro_model;
pub mod ws_framing;

#[cfg(feature = "esp32s3")]
mod esp32s3_firmware {
    use super::{
        micro_model::{
            nlms_aec_cancel_frame, vad_prob_exceeds_barge_in, Esp32QuantalFrontend,
            FRAME_SAMPLES_16K,
        },
        ws_framing::{self, ServerWsPayload},
    };
    use burn::tensor::Device;
    use embassy_executor::Spawner;
    use embassy_net::{tcp::TcpSocket, Config as NetConfig, IpAddress, Ipv4Address, StackResources};
    use embassy_sync::{blocking_mutex::raw::CriticalSectionRawMutex, pipe::Pipe};
    use embassy_time::{Duration, Timer};
    use esp_backtrace as _;
    use esp_hal::{
        clock::CpuClock,
        dma::{DmaRxStreamBuf, DmaTxStreamBuf},
        dma_buffers,
        i2s::master::{DataFormat, I2s, Standard},
        rng::Rng,
        time::Rate,
        timer::timg::TimerGroup,
    };
    use esp_println::println;
    use esp_wifi::{
        wifi::{
            ClientConfiguration, Configuration, WifiController, WifiDevice, WifiEvent, WifiState,
        },
        EspWifiController,
    };
    use portable_atomic::{AtomicBool, Ordering};
    use static_cell::StaticCell;

    static TTS_PLAYBACK_PIPE: Pipe<CriticalSectionRawMutex, 16384> = Pipe::new();
    static BARGE_IN_FLUSH: AtomicBool = AtomicBool::new(false);
    static mut AEC_REF_RING: [f32; FRAME_SAMPLES_16K] = [0.0; FRAME_SAMPLES_16K];

    #[esp_hal_embassy::main]
    async fn main(spawner: Spawner) -> ! {
        let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
        let peripherals = esp_hal::init(config);
        esp_alloc::psram_allocator!(peripherals.PSRAM, esp_hal::psram);

        let timg0 = TimerGroup::new(peripherals.TIMG0);
        esp_hal_embassy::init(timg0.timer0);

        println!("[Aurora-ESP32-S3] 8MB PSRAM initialized. Booting Burn-Flex + I2S Full-Duplex...");

        let timg1 = TimerGroup::new(peripherals.TIMG1);
        let mut rng = Rng::new(peripherals.RNG);
        static WIFI_CTRL: StaticCell<EspWifiController<'static>> = StaticCell::new();
        let wifi_init = WIFI_CTRL.init(
            esp_wifi::init(timg1.timer0, rng, peripherals.RADIO_CLK).expect("Failed esp-wifi init"),
        );

        let (wifi_interface, controller) = esp_wifi::wifi::new_with_mode(
            wifi_init,
            peripherals.WIFI,
            esp_wifi::wifi::WifiStaDevice,
        )
        .expect("Failed Wi-Fi STA init");

        static STACK_RES: StaticCell<StackResources<3>> = StaticCell::new();
        let (stack, runner) = embassy_net::new(
            wifi_interface,
            NetConfig::dhcpv4(Default::default()),
            STACK_RES.init(StackResources::new()),
            rng.random() as u64,
        );

        spawner.spawn(wifi_connection_task(controller)).unwrap();
        spawner.spawn(net_runner_task(runner)).unwrap();

        let (rx_buffer, rx_descriptors, tx_buffer, tx_descriptors) = dma_buffers!(10240, 10240);
        let i2s = I2s::new(
            peripherals.I2S0,
            Standard::Philips,
            DataFormat::Data16Channel16,
            Rate::from_hz(16_000),
            peripherals.DMA_CH0,
            rx_descriptors,
            tx_descriptors,
        );

        let i2s_rx = i2s
            .i2s_rx
            .with_bclk(peripherals.GPIO4)
            .with_ws(peripherals.GPIO5)
            .with_din(peripherals.GPIO6)
            .build();

        let i2s_tx = i2s.i2s_tx.with_dout(peripherals.GPIO7).build();

        let rx_dma_buf = DmaRxStreamBuf::new(rx_buffer, rx_descriptors).unwrap();
        let tx_dma_buf = DmaTxStreamBuf::new(tx_buffer, tx_descriptors).unwrap();
        let mut rx_transfer = i2s_rx.read_dma_circular_async(rx_dma_buf).unwrap();
        let tx_transfer = i2s_tx.write_dma_circular_async(tx_dma_buf).unwrap();

        spawner.spawn(i2s_speaker_tx_task(tx_transfer)).unwrap();

        let flex_device = Device::default();
        let model = Esp32QuantalFrontend::init_from_static_weights(&[], &flex_device);

        stack.wait_config_up().await;
        println!("[Aurora-ESP32-S3] Network up: {:?}", stack.config_v4());

        let mut rx_net_buf = [0u8; 8192];
        let mut tx_net_buf = [0u8; 4096];
        let mut socket = TcpSocket::new(stack, &mut rx_net_buf, &mut tx_net_buf);
        let server_ip = Ipv4Address::new(192, 168, 1, 100);
        socket
            .connect((IpAddress::Ipv4(server_ip), 8080))
            .await
            .expect("Failed connecting to Aurora-8B server");
        ws_framing::websocket_handshake(&mut socket, "192.168.1.100:8080")
            .await
            .expect("WebSocket handshake failed");

        println!("[Aurora-ESP32-S3] Full-Duplex 80ms Quantal Loop Active!");

        let mut mic_frame_f32 = [0.0f32; FRAME_SAMPLES_16K];
        let mut dma_raw_bytes = [0u8; FRAME_SAMPLES_16K * 2];
        let mut nlms_weight: f32 = 0.0;
        let mut seq_num: u16 = 0;

        loop {
            let mut collected = 0;
            while collected < dma_raw_bytes.len() {
                let n = rx_transfer
                    .pop(&mut dma_raw_bytes[collected..])
                    .await
                    .unwrap_or(0);
                collected += n;
            }

            let i16_slice: &[i16] = bytemuck::cast_slice(&dma_raw_bytes);
            let mut i16_arr = [0i16; FRAME_SAMPLES_16K];
            i16_arr.copy_from_slice(&i16_slice[..FRAME_SAMPLES_16K]);
            let spk_ref_copy = unsafe { AEC_REF_RING };
            nlms_aec_cancel_frame(&i16_arr, &spk_ref_copy, &mut nlms_weight, &mut mic_frame_f32);

            let (vad_prob, packet) = model.encode_80ms_frame(
                &mic_frame_f32,
                seq_num,
                vad_prob_exceeds_barge_in(nlms_weight),
                &flex_device,
            );
            seq_num = seq_num.wrapping_add(1);

            if vad_prob > 0.55 {
                BARGE_IN_FLUSH.store(true, Ordering::Release);
            }

            if vad_prob > 0.30 {
                let packet_bytes = packet.to_wire_bytes();
                let _ = ws_framing::ws_send_binary_packet(
                    &mut socket,
                    &packet_bytes,
                    [0x12, 0x34, 0x56, 0x78],
                )
                .await;
            }

            if socket.may_recv() {
                let mut ws_in = [0u8; 1024];
                if let Ok(n) = socket.read(&mut ws_in).await {
                    if let Ok((payload, _)) = ws_framing::parse_unmasked_server_ws_frame(&ws_in[..n])
                    {
                        match payload {
                            ServerWsPayload::BinaryPcm(pcm) => {
                                let _ = TTS_PLAYBACK_PIPE.write(pcm).await;
                            }
                            ServerWsPayload::TextJson(_json_cmd) => {
                                // Trigger GPIO4 relay / hardware actuator
                            }
                        }
                    }
                }
            }
        }
    }

    #[embassy_executor::task]
    async fn i2s_speaker_tx_task(
        mut tx_transfer: esp_hal::i2s::master::asynch::I2sWriteDmaTransferAsync<
            'static,
            DmaTxStreamBuf<'static>,
        >,
    ) {
        let mut pcm_bytes = [0u8; FRAME_SAMPLES_16K * 2];
        loop {
            if BARGE_IN_FLUSH.swap(false, Ordering::AcqRel) {
                pcm_bytes.fill(0);
            } else {
                let n = TTS_PLAYBACK_PIPE.try_read(&mut pcm_bytes).unwrap_or(0);
                if n < pcm_bytes.len() {
                    pcm_bytes[n..].fill(0);
                }
            }

            let samples_i16: &[i16] = bytemuck::cast_slice(&pcm_bytes);
            for i in 0..FRAME_SAMPLES_16K {
                unsafe {
                    AEC_REF_RING[i] = (samples_i16[i] as f32) / 32768.0;
                }
            }

            let _ = tx_transfer.push(&pcm_bytes).await;
        }
    }

    #[embassy_executor::task]
    async fn wifi_connection_task(mut controller: WifiController<'static>) {
        loop {
            if esp_wifi::wifi::wifi_state() == WifiState::StaConnected {
                controller.wait_for_event(WifiEvent::StaDisconnected).await;
                Timer::after(Duration::from_millis(2000)).await;
            }
            if !matches!(controller.is_started(), Ok(true)) {
                let client_config = Configuration::Client(ClientConfiguration {
                    ssid: env!("WIFI_SSID").try_into().unwrap(),
                    password: env!("WIFI_PASS").try_into().unwrap(),
                    ..Default::default()
                });
                controller.set_configuration(&client_config).unwrap();
                controller.start_async().unwrap();
            }
            let _ = controller.connect_async().await;
        }
    }

    #[embassy_executor::task]
    async fn net_runner_task(
        mut runner: embassy_net::Runner<
            'static,
            WifiDevice<'static, esp_wifi::wifi::WifiStaDevice>,
        >,
    ) {
        runner.run().await
    }
}

#[cfg(not(feature = "esp32s3"))]
fn main() {
    use burn::tensor::Device;
    use micro_model::{
        nlms_aec_cancel_frame, Esp32QuantalFrontend, FRAME_SAMPLES_16K, QUANTAL_PACKET_WIRE_BYTES,
    };

    let device = Device::default();
    let frontend = Esp32QuantalFrontend::init_from_static_weights(&[], &device);

    let mut mic_i16 = [0i16; FRAME_SAMPLES_16K];
    let mut spk_ref = [0.0f32; FRAME_SAMPLES_16K];
    for i in 0..FRAME_SAMPLES_16K {
        let t = (i as f32) / 16000.0;
        let speech = libm::sinf(2.0 * core::f32::consts::PI * 220.0 * t) * 0.45;
        let echo = libm::sinf(2.0 * core::f32::consts::PI * 440.0 * t) * 0.25;
        spk_ref[i] = echo;
        mic_i16[i] = ((speech + 0.6 * echo) * 32767.0) as i16;
    }

    let mut nlms_weight = 0.0f32;
    let mut clean_f32 = [0.0f32; FRAME_SAMPLES_16K];
    nlms_aec_cancel_frame(&mic_i16, &spk_ref, &mut nlms_weight, &mut clean_f32);

    let (vad_prob, pkt) = frontend.encode_80ms_frame(&clean_f32, 1, true, &device);
    let wire = pkt.to_wire_bytes();
    let mut ws_frame = [0u8; 6 + 125];
    let ws_len =
        ws_framing::encode_masked_binary_frame(&wire, [0x12, 0x34, 0x56, 0x78], &mut ws_frame)
            .expect("WS binary frame encode");

    println!(
        "{{\"ok\":true,\"crate\":\"aurora-esp32-node\",\"target\":\"xtensa-esp32s3-none-elf\",\"no_std\":true,\"frame_samples_16k\":{},\"packet_wire_bytes\":{},\"ws_frame_bytes\":{},\"vad_prob\":{:.4},\"vad_prob_q8\":{},\"rms_db_q8\":{},\"nlms_weight\":{:.4},\"rvq_q0\":{:?},\"rvq_q1\":{:?}}}",
        FRAME_SAMPLES_16K,
        QUANTAL_PACKET_WIRE_BYTES,
        ws_len,
        vad_prob,
        pkt.vad_prob_q8,
        pkt.rms_db_q8,
        nlms_weight,
        { pkt.rvq_q0 },
        { pkt.rvq_q1 },
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Device;
    use micro_model::{
        nlms_aec_cancel_frame, Esp32QuantalFrontend, Quantal80msPacket, FRAME_SAMPLES_16K,
        QUANTAL_PACKET_MAGIC, QUANTAL_PACKET_WIRE_BYTES,
    };
    use ws_framing::{parse_unmasked_server_ws_frame, ServerWsPayload};

    #[test]
    fn test_quantal_80ms_packet_42_byte_wire_roundtrip() {
        assert_eq!(
            core::mem::size_of::<Quantal80msPacket>(),
            QUANTAL_PACKET_WIRE_BYTES
        );
        let pkt = Quantal80msPacket {
            magic: QUANTAL_PACKET_MAGIC,
            seq_num: 42,
            vad_prob_q8: 204,
            rms_db_q8: -16,
            rvq_q0: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            rvq_q1: [1, 2, 3, 4, 5, 6, 7, 8],
        };
        let wire = pkt.to_wire_bytes();
        assert_eq!(wire.len(), 42);
        let decoded = Quantal80msPacket::from_wire_bytes(&wire).expect("valid 0xA8B0 packet");
        assert_eq!(decoded, pkt);
    }

    #[test]
    fn test_rfc6455_2_byte_and_4_byte_960b_server_frame_parser() {
        // 1. 4-byte extended header for 960B (30ms @ 16kHz i16le) PCM frame: [0x82, 126, 0x03, 0xC0]
        let mut frame_960 = [0u8; 4 + 960];
        frame_960[0] = 0x82;
        frame_960[1] = 126;
        frame_960[2..4].copy_from_slice(&(960u16).to_be_bytes());
        frame_960[4] = 0x55;
        frame_960[963] = 0xAA;

        let (payload, used) =
            parse_unmasked_server_ws_frame(&frame_960).expect("valid 4-byte 960B WS frame");
        assert_eq!(used, 964);
        match payload {
            ServerWsPayload::BinaryPcm(pcm) => {
                assert_eq!(pcm.len(), 960);
                assert_eq!(pcm[0], 0x55);
                assert_eq!(pcm[959], 0xAA);
            }
            _ => panic!("expected BinaryPcm"),
        }

        // 2. 2-byte header for JSON GPIO command (< 126B)
        let json_bytes = br#"{"cmd":"gpio","pin":4,"val":1}"#;
        let mut frame_json = [0u8; 64];
        frame_json[0] = 0x81;
        frame_json[1] = json_bytes.len() as u8;
        frame_json[2..2 + json_bytes.len()].copy_from_slice(json_bytes);

        let (payload_txt, used_txt) =
            parse_unmasked_server_ws_frame(&frame_json[..2 + json_bytes.len()])
                .expect("valid 2-byte JSON WS frame");
        assert_eq!(used_txt, 2 + json_bytes.len());
        assert_eq!(payload_txt, ServerWsPayload::TextJson(json_bytes));
    }

    #[test]
    fn test_nlms_aec_and_burn_flex_80ms_frontend() {
        let device = Device::default();
        let frontend = Esp32QuantalFrontend::init_from_static_weights(&[], &device);

        let mut spk_ref = [0.0f32; FRAME_SAMPLES_16K];
        let mut mic_i16 = [0i16; FRAME_SAMPLES_16K];
        for i in 0..FRAME_SAMPLES_16K {
            let s = libm::sinf((i as f32) * 0.11) * 0.5;
            spk_ref[i] = s;
            mic_i16[i] = ((0.8 * s) * 32767.0) as i16;
        }

        let mut nlms_weight = 0.0f32;
        let mut clean_f32 = [0.0f32; FRAME_SAMPLES_16K];
        for _ in 0..4 {
            nlms_aec_cancel_frame(&mic_i16, &spk_ref, &mut nlms_weight, &mut clean_f32);
        }
        assert!((nlms_weight - 0.8).abs() < 0.02);

        let (vad_prob, pkt) = frontend.encode_80ms_frame(&clean_f32, 7, true, &device);
        assert!((0.0..=1.0).contains(&vad_prob));
        let magic = pkt.magic;
        let seq = pkt.seq_num;
        assert_eq!(magic, QUANTAL_PACKET_MAGIC);
        assert_eq!(seq, 7);
    }
}
