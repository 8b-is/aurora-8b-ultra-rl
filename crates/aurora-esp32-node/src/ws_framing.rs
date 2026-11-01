//! Zero-allocation `#![no_std]` RFC-6455 Binary & Text WebSocket Framing for `embassy-net`
//! and host verification. Supports both 2-byte (`< 126B`) and 4-byte (`126..=960B`) server headers.

#![allow(clippy::result_unit_err)]

pub const WS_UPGRADE_REQ: &[u8] = b"GET /v1/aurora-esp32 HTTP/1.1\r\n\
Host: aurora.local\r\n\
Upgrade: websocket\r\n\
Connection: Upgrade\r\n\
Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\
Sec-WebSocket-Version: 13\r\n\r\n";

/// Parsed unmasked server-to-ESP32 RFC-6455 frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerWsPayload<'a> {
    /// `0x82`: Binary `16 kHz` `i16le` PCM chunk (`128..=960` bytes).
    BinaryPcm(&'a [u8]),
    /// `0x81`: Text JSON GPIO / hardware command (`{"cmd":"gpio","pin":4,"val":1}`).
    TextJson(&'a [u8]),
}

/// Verifies that an HTTP/1.1 server response begins with a `101` Switching Protocols status code.
pub fn validate_ws_upgrade_response(resp_buf: &[u8]) -> Result<(), ()> {
    if resp_buf.len() < 12 || &resp_buf[9..12] != b"101" {
        return Err(());
    }
    Ok(())
}

/// Encodes a client-to-server masked binary WebSocket frame (`Opcode 0x82`, `len <= 125`)
/// into `out_frame` and returns the total frame length (`6 + payload.len()`).
pub fn encode_masked_binary_frame(
    payload: &[u8],
    mask_key: [u8; 4],
    out_frame: &mut [u8; 6 + 125],
) -> Result<usize, ()> {
    let len = payload.len();
    if len > 125 {
        return Err(());
    }
    out_frame[0] = 0x82; // FIN = 1, Opcode = 0x2 (Binary)
    out_frame[1] = 0x80 | (len as u8); // MASK = 1 | payload_len
    out_frame[2..6].copy_from_slice(&mask_key);

    for i in 0..len {
        out_frame[6 + i] = payload[i] ^ mask_key[i & 3];
    }
    Ok(6 + len)
}

/// Decodes a masked client-to-server binary WebSocket frame (`Opcode 0x82`, `len <= 125`)
/// back into `out_payload` and returns the unmasked payload length.
pub fn decode_masked_binary_frame(
    frame: &[u8],
    out_payload: &mut [u8; 125],
) -> Result<usize, ()> {
    if frame.len() < 6 || frame[0] != 0x82 || (frame[1] & 0x80) == 0 {
        return Err(());
    }
    let len = (frame[1] & 0x7F) as usize;
    if len > 125 || frame.len() < 6 + len {
        return Err(());
    }
    let mask_key = [frame[2], frame[3], frame[4], frame[5]];
    for i in 0..len {
        out_payload[i] = frame[6 + i] ^ mask_key[i & 3];
    }
    Ok(len)
}

/// Zero-heap parser for unmasked server-to-ESP32 RFC-6455 frames supporting both:
/// - 2-byte header (`len_code < 126`, e.g. JSON GPIO control frames)
/// - 4-byte extended header (`len_code == 126`, e.g. `128..=960` byte PCM frames)
pub fn parse_unmasked_server_ws_frame(rx_buf: &[u8]) -> Result<(ServerWsPayload<'_>, usize), ()> {
    if rx_buf.len() < 2 {
        return Err(());
    }
    let opcode = rx_buf[0] & 0x0F;
    let is_masked = (rx_buf[1] & 0x80) != 0;
    if is_masked {
        return Err(());
    }
    let len_code = (rx_buf[1] & 0x7F) as usize;
    let (header_len, payload_len) = if len_code < 126 {
        (2usize, len_code)
    } else if len_code == 126 {
        if rx_buf.len() < 4 {
            return Err(());
        }
        let ext_len = u16::from_be_bytes([rx_buf[2], rx_buf[3]]) as usize;
        (4usize, ext_len)
    } else {
        return Err(());
    };

    let total_frame_len = header_len + payload_len;
    if rx_buf.len() < total_frame_len {
        return Err(());
    }
    let slice = &rx_buf[header_len..total_frame_len];
    match opcode {
        0x01 => Ok((ServerWsPayload::TextJson(slice), total_frame_len)),
        0x02 => Ok((ServerWsPayload::BinaryPcm(slice), total_frame_len)),
        _ => Err(()),
    }
}

#[cfg(feature = "esp32s3")]
use embassy_net::tcp::TcpSocket;

#[cfg(feature = "esp32s3")]
pub async fn websocket_handshake(socket: &mut TcpSocket<'_>, host: &str) -> Result<(), ()> {
    let _ = host;
    socket.write(WS_UPGRADE_REQ).await.map_err(|_| ())?;

    let mut resp_buf = [0u8; 256];
    let n = socket.read(&mut resp_buf).await.map_err(|_| ())?;
    validate_ws_upgrade_response(&resp_buf[..n])
}

#[cfg(feature = "esp32s3")]
pub async fn ws_send_binary_packet(
    socket: &mut TcpSocket<'_>,
    payload: &[u8],
    mask_key: [u8; 4],
) -> Result<(), ()> {
    let mut frame = [0u8; 6 + 125];
    let total_len = encode_masked_binary_frame(payload, mask_key, &mut frame)?;

    let mut written = 0;
    while written < total_len {
        let w = socket
            .write(&frame[written..total_len])
            .await
            .map_err(|_| ())?;
        written += w;
    }
    Ok(())
}
