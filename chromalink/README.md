# AI ChromaLink

Spectral Color-Block Mirroring for Secure, Infrastructure-Free Optical Communication

BuildWithAI / INNOVATEX proof-of-concept with two laptops:
- TX laptop shows 8 RGB blocks on screen.
- RX laptop camera reads those 8 blocks and mirrors/decodes them in real time.

No internet, no radio, no wires.

## System Architecture

- `chromalink/tx/tx_app.py`: Transmitter GUI (8-block display + message encoder)
- `chromalink/rx/rx_app.py`: Receiver GUI (camera + AI/CV detection + decode)
- `chromalink/core/encoder.py`: Optical framing protocol (SYNC start/end + char frames + gap frames)
- `chromalink/core/detector.py`: CV pipeline (screen detect, perspective normalization, fallback strip, smoothing)

## Protocol Summary

- 8 blocks represent one payload frame.
- Bit `1` = signal color (ASCII-dependent hue), bit `0` = white.
- `SYNC_START` = red marker in block 0.
- `SYNC_END` = blue marker in block 0.
- `GAP` frames (dark blocks) are inserted between characters for stable RX decoding and repeated-letter support.

## Prerequisites (Both Laptops)

```bash
pip install -r chromalink/requirements.txt
```

Python 3.11+ recommended.

## Run Mode For Your Setup

You said this machine is RX and another laptop is TX.

### 1) On TX laptop

```bash
python chromalink/tx/tx_app.py
```

### 2) On this RX laptop

```bash
python chromalink/rx/rx_app.py
```

## Live Demo Procedure

1. Put both laptops facing each other (RX camera looking at TX screen).
2. On RX app:
- Select camera index (`Cam 0`, `Cam 1`, or `Cam 2`)
- Click `START DETECTION`
- Use `AUTO` mode first, switch to `FALLBACK` if needed
3. On TX app:
- Type a short message (for example `HELLO`)
- Click `TRANSMIT`
4. RX should:
- Mirror the 8-block pattern
- Decode and show text in `DECODED MESSAGE`

## Practical Demo Settings

- TX speed: start around `0.8s/char`
- If decoding is unstable at distance: increase TX speed to `1.2s` to `1.5s/char`
- Keep TX screen brightness high and reduce strong backlight behind TX

## Troubleshooting

- RX not finding screen: switch RX detection mode to `FALLBACK`
- Camera unavailable: try different camera index
- Wrong characters: increase TX speed and improve alignment
- Flicker/noise: reduce ambient glare and keep camera focus locked

## Notes For Pitch

- Emphasize that this is real optical data transmission, not only visual mirroring.
- Mention AI/CV robustness: contour detection, perspective correction, lighting normalization, temporal smoothing.
- Highlight use cases: defense, disaster response, remote hilly regions with no infrastructure.
