//! A minimal PNG writer: 8-bit or 16-bit grayscale, zlib stream of stored
//! (uncompressed) deflate blocks, so no compressor is needed.

fn crc32(data: &[u8]) -> u32 {
    let mut table = [0u32; 256];
    for (n, t) in table.iter_mut().enumerate() {
        let mut c = n as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
        }
        *t = c;
    }
    let mut c = 0xFFFF_FFFFu32;
    for &b in data {
        c = table[((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8);
    }
    c ^ 0xFFFF_FFFF
}

fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for chunk in data.chunks(5552) {
        for &x in chunk {
            a += x as u32;
            b += a;
        }
        a %= 65521;
        b %= 65521;
    }
    (b << 16) | a
}

fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend((data.len() as u32).to_be_bytes());
    let mut c = kind.to_vec();
    c.extend_from_slice(data);
    out.extend_from_slice(&c);
    out.extend(crc32(&c).to_be_bytes());
}

/// Encode a grayscale image (bit depth 8 or 16; 16-bit samples big-endian in `px` order).
pub fn encode_gray(w: usize, h: usize, depth: u8, px: &[u8]) -> Vec<u8> {
    let bpp = if depth == 16 { 2 } else { 1 };
    assert_eq!(px.len(), w * h * bpp);
    let mut raw = Vec::with_capacity((w * bpp + 1) * h);
    for y in 0..h {
        raw.push(0); // filter: none
        raw.extend_from_slice(&px[y * w * bpp..(y + 1) * w * bpp]);
    }
    let mut z = vec![0x78, 0x01];
    let n = raw.len();
    let mut pos = 0;
    while pos < n {
        let len = (n - pos).min(65535);
        z.push(if pos + len == n { 1 } else { 0 });
        z.extend((len as u16).to_le_bytes());
        z.extend((!(len as u16)).to_le_bytes());
        z.extend_from_slice(&raw[pos..pos + len]);
        pos += len;
    }
    z.extend(adler32(&raw).to_be_bytes());
    let mut out = vec![137, 80, 78, 71, 13, 10, 26, 10];
    let mut ihdr = Vec::new();
    ihdr.extend((w as u32).to_be_bytes());
    ihdr.extend((h as u32).to_be_bytes());
    ihdr.extend([depth, 0, 0, 0, 0]);
    chunk(&mut out, b"IHDR", &ihdr);
    chunk(&mut out, b"IDAT", &z);
    chunk(&mut out, b"IEND", &[]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_checksums() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
        let png = encode_gray(3, 2, 8, &[1, 2, 3, 4, 5, 6]);
        assert_eq!(&png[1..4], b"PNG");
    }
}
