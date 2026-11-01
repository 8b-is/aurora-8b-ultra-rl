pub mod esp32_engine;
pub mod protocol;

pub use esp32_engine::{
    Esp32DeviceSession, Esp32GatewayEngine, Esp32QuantalAdapter, Esp32StepOutcome,
};
pub use protocol::{
    chunk_esp32_ws_pcm_frames, kokoro_24k_to_esp32_ws_chunks, resample_24k_to_16k_i16,
    DecodedEsp32Frame, PacketDecodeError, Quantal80msPacket, ESP32_WS_MAX_CHUNK_BYTES,
    ESP32_WS_MIN_CHUNK_BYTES, QUANTAL_MAGIC_A8B0, QUANTAL_PACKET_BYTES,
};

use std::{net::SocketAddr, sync::Arc, time::Duration};

use axum::{
    extract::{
        ws::{Message as WsMessage, WebSocket, WebSocketUpgrade},
        State,
    },
    http::HeaderMap,
    response::IntoResponse,
    routing::get,
    Router,
};
use futures_util::{SinkExt, StreamExt};
use tokio::sync::{mpsc, Mutex as AsyncMutex};
use tokio_util::sync::CancellationToken;

/// Build the Axum router serving `/v1/aurora-esp32`.
pub fn build_esp32_router(engine: Arc<Esp32GatewayEngine>) -> Router {
    Router::new()
        .route("/v1/aurora-esp32", get(esp32_ws_upgrade))
        .with_state(engine)
}

pub async fn esp32_ws_upgrade(
    ws: WebSocketUpgrade,
    headers: HeaderMap,
    State(engine): State<Arc<Esp32GatewayEngine>>,
) -> impl IntoResponse {
    let peer_str = headers
        .get("x-forwarded-for")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("127.0.0.1:8080");
    let peer: SocketAddr = peer_str
        .parse()
        .unwrap_or_else(|_| SocketAddr::from(([127, 0, 0, 1], 8080)));
    ws.on_upgrade(move |socket| handle_esp32_socket(socket, peer, engine))
}

async fn handle_esp32_socket(
    socket: WebSocket,
    peer: SocketAddr,
    engine: Arc<Esp32GatewayEngine>,
) {
    let (ws_sender, mut ws_receiver) = socket.split();
    let ws_sender = Arc::new(AsyncMutex::new(ws_sender));

    let (tts_tx, mut tts_rx) =
        mpsc::unbounded_channel::<(Arc<Vec<f32>>, Option<String>, CancellationToken)>();
    let mut active_cancel: Option<CancellationToken> = None;

    // Egress Pacer: streams 960B (30ms) 16kHz i16le chunks padded to >= 128B
    let sender_clone = Arc::clone(&ws_sender);
    let egress_task = tokio::spawn(async move {
        while let Some((pcm_24k, gpio_json, cancel_tok)) = tts_rx.recv().await {
            if let Some(cmd) = gpio_json {
                let mut guard = sender_clone.lock().await;
                let _ = guard.send(WsMessage::Text(cmd.into())).await;
            }
            for chunk in kokoro_24k_to_esp32_ws_chunks(&pcm_24k) {
                if cancel_tok.is_cancelled() {
                    break;
                }
                {
                    let mut guard = sender_clone.lock().await;
                    if guard.send(WsMessage::Binary(chunk.into())).await.is_err() {
                        return;
                    }
                }
                tokio::select! {
                    _ = cancel_tok.cancelled() => break,
                    _ = tokio::time::sleep(Duration::from_millis(25)) => {}
                }
            }
        }
    });

    while let Some(Ok(msg)) = ws_receiver.next().await {
        match msg {
            WsMessage::Binary(bytes) => {
                if let Ok(frame) = Quantal80msPacket::decode_from_slice(&bytes) {
                    let tts_active = active_cancel
                        .as_ref()
                        .map(|t| !t.is_cancelled())
                        .unwrap_or(false);
                    match engine.step_80ms_packet(peer, frame, tts_active) {
                        Esp32StepOutcome::BargeInInterrupt => {
                            if let Some(tok) = active_cancel.take() {
                                tok.cancel();
                            }
                        }
                        Esp32StepOutcome::Listening { .. } => {}
                        Esp32StepOutcome::TurnCommitted {
                            gpio_json,
                            kokoro_pcm_24k,
                            ..
                        } => {
                            if let Some(tok) = active_cancel.take() {
                                tok.cancel();
                            }
                            let tok = CancellationToken::new();
                            active_cancel = Some(tok.clone());
                            let _ = tts_tx.send((kokoro_pcm_24k, gpio_json, tok));
                        }
                    }
                }
            }
            WsMessage::Close(_) => break,
            _ => {}
        }
    }

    egress_task.abort();
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Device;

    #[test]
    fn test_esp32_gateway_engine_thinking_pause_and_commit() {
        let device = Device::default();
        let engine = Esp32GatewayEngine::new(device);
        let peer: SocketAddr = "192.168.1.55:9001".parse().unwrap();

        let speech_pkt = Quantal80msPacket::new(1, 220, -14, [12; 10], [7; 8]);
        let speech_frame = Quantal80msPacket::decode_from_slice(speech_pkt.as_bytes()).unwrap();
        let out1 = engine.step_80ms_packet(peer, speech_frame, false);
        assert!(matches!(out1, Esp32StepOutcome::Listening { seq: 1 }));

        // 4 silent frames (< 5 ticks) -> Thinking pause holds turn
        for seq in 2..=5 {
            let sil_pkt = Quantal80msPacket::new(seq, 10, -55, [0; 10], [0; 8]);
            let sil_frame = Quantal80msPacket::decode_from_slice(sil_pkt.as_bytes()).unwrap();
            let out = engine.step_80ms_packet(peer, sil_frame, false);
            assert!(matches!(out, Esp32StepOutcome::Listening { .. }));
        }

        // 5th silent frame -> TurnCommitted with Kokoro 24kHz audio + GPIO JSON
        let commit_pkt = Quantal80msPacket::new(6, 10, -55, [0; 10], [0; 8]);
        let commit_frame = Quantal80msPacket::decode_from_slice(commit_pkt.as_bytes()).unwrap();
        let out_commit = engine.step_80ms_packet(peer, commit_frame, false);
        match out_commit {
            Esp32StepOutcome::TurnCommitted {
                kokoro_pcm_24k,
                gpio_json,
                ..
            } => {
                assert!(gpio_json.is_some());
                let chunks = kokoro_24k_to_esp32_ws_chunks(&kokoro_pcm_24k);
                assert!(!chunks.is_empty());
                for c in chunks {
                    assert!(c.len() >= 128 && c.len() <= 960);
                }
            }
            other => panic!("Expected TurnCommitted, got {other:?}"),
        }
    }
}
