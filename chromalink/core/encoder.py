"""
AI ChromaLink protocol encoder/decoder.

The system transmits one character per frame over 8 color blocks:
- bit = 1: signal color (derived from character code hue)
- bit = 0: white

Framing:
- SYNC_START frame: first block bright red
- SYNC_END frame: first block bright blue
- GAP frame: all blocks dark (idle color), inserted between characters
"""

from __future__ import annotations

import colorsys
from typing import List, Sequence, Tuple

RGB = Tuple[int, int, int]
Frame = List[RGB]

WHITE: RGB = (255, 255, 255)
BLACK: RGB = (0, 0, 0)
SYNC_START: RGB = (255, 20, 20)
SYNC_END: RGB = (20, 20, 255)
IDLE_COLOR: RGB = (30, 30, 30)

SATURATION = 0.92
VALUE = 0.95
NUM_BLOCKS = 8
ASCII_RANGE = 128


def ascii_to_rgb(char_code: int) -> RGB:
    """Map an ASCII code (0-127) to a distinct RGB color via HSV hue."""
    code = int(char_code) & 0x7F
    hue = (code / float(ASCII_RANGE)) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, SATURATION, VALUE)
    return int(r * 255), int(g * 255), int(b * 255)


def _is_white_like(rgb: RGB) -> bool:
    r, g, b = rgb
    return r > 220 and g > 220 and b > 220


def _is_signal(rgb: RGB) -> bool:
    if _is_white_like(rgb):
        return False
    r, g, b = rgb
    _, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    return s > 0.30 and v > 0.25


def rgb_to_ascii(rgb: RGB) -> int:
    """
    Reverse-map an RGB color back to ASCII using nearest hue match.
    Returns -1 for non-signal colors.
    """
    if not _is_signal(rgb):
        return -1

    r, g, b = rgb
    h, _, _ = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)

    best_code = -1
    best_dist = float("inf")
    for code in range(ASCII_RANGE):
        expected_hue = (code / float(ASCII_RANGE)) % 1.0
        dist = min(abs(h - expected_hue), 1.0 - abs(h - expected_hue))
        if dist < best_dist:
            best_dist = dist
            best_code = code

    # Reject outliers: roughly 2.5 hue bins.
    if best_dist > (1.0 / ASCII_RANGE) * 2.5:
        return -1
    return best_code


def encode_character(char: str) -> Frame:
    """
    Encode one character into 8 color blocks.
    Each block carries one bit of the 7-bit ASCII code (LSB at block 0).
    """
    if not char:
        return [WHITE] * NUM_BLOCKS

    code = ord(char[0]) & 0x7F
    signal_color = ascii_to_rgb(code)
    blocks: Frame = []
    for bit_pos in range(NUM_BLOCKS):
        bit = (code >> bit_pos) & 1
        blocks.append(signal_color if bit else WHITE)
    return blocks


def decode_blocks(block_colors: Sequence[RGB]) -> str:
    """
    Decode one 8-block frame into a character.
    Returns:
    - '\\x02' for SYNC_START
    - '\\x03' for SYNC_END
    - '' for gap/idle/invalid
    - printable ASCII character for valid payload frames
    """
    if len(block_colors) != NUM_BLOCKS:
        return ""

    if is_sync_start(block_colors):
        return "\x02"
    if is_sync_end(block_colors):
        return "\x03"
    if is_idle(block_colors):
        return ""

    reconstructed_bits = 0
    strongest_signal: RGB | None = None
    strongest_sat = 0.0

    for i, rgb in enumerate(block_colors):
        if not _is_signal(rgb):
            continue
        reconstructed_bits |= 1 << i

        r, g, b = rgb
        _, sat, _ = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
        if sat > strongest_sat:
            strongest_sat = sat
            strongest_signal = rgb

    if strongest_signal is None or reconstructed_bits == 0:
        return ""

    bit_code = reconstructed_bits & 0x7F
    hue_code = rgb_to_ascii(strongest_signal)
    char_code = bit_code

    # If bit reconstruction and hue reconstruction disagree heavily, trust hue.
    if hue_code != -1 and _hamming_distance(bit_code, hue_code) >= 3:
        char_code = hue_code

    if 32 <= char_code <= 126:
        return chr(char_code)
    return ""


def _hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def encode_message(text: str, include_gap_frames: bool = True) -> list[tuple[str, Frame]]:
    """
    Encode text into a transmission sequence.

    Returns:
    [SYNC_START] [CHAR:x] [GAP] [CHAR:y] ... [SYNC_END]
    GAP frames are inserted between characters to enable robust RX framing.
    """
    frames: list[tuple[str, Frame]] = []
    frames.append(("SYNC_START", get_sync_start_frame()))

    for index, char in enumerate(text):
        frames.append((f"CHAR:{char}", encode_character(char)))
        if include_gap_frames and index < len(text) - 1:
            frames.append(("GAP", get_gap_frame()))

    frames.append(("SYNC_END", get_sync_end_frame()))
    return frames


def get_idle_frame() -> Frame:
    """Frame shown while TX is idle/standby."""
    return [IDLE_COLOR] * NUM_BLOCKS


def get_gap_frame() -> Frame:
    """Inter-character separator frame."""
    return [IDLE_COLOR] * NUM_BLOCKS


def get_sync_start_frame() -> Frame:
    return [SYNC_START] + [WHITE] * (NUM_BLOCKS - 1)


def get_sync_end_frame() -> Frame:
    return [SYNC_END] + [WHITE] * (NUM_BLOCKS - 1)


def is_sync_start(block_colors: Sequence[RGB]) -> bool:
    if not block_colors:
        return False
    r, g, b = block_colors[0]
    return r > 200 and g < 70 and b < 70


def is_sync_end(block_colors: Sequence[RGB]) -> bool:
    if not block_colors:
        return False
    r, g, b = block_colors[0]
    return r < 70 and g < 70 and b > 200


def is_idle(block_colors: Sequence[RGB]) -> bool:
    """Check if every block is dark enough to be considered idle/gap."""
    if len(block_colors) != NUM_BLOCKS:
        return False
    for r, g, b in block_colors:
        if r > 70 or g > 70 or b > 70:
            return False
    return True
