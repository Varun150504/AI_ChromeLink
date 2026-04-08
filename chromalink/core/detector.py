"""Computer-vision detector for AI ChromaLink RX."""

from __future__ import annotations

from collections import deque
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

RGB = Tuple[int, int, int]

BLOCK_SAMPLE_RATIO = 0.45
TEMPORAL_WINDOW = 4
MIN_CONTOUR_AREA = 3000
NUM_BLOCKS = 8


class ColorDetector:
    """Detect 8 TX color blocks from a camera frame."""

    def __init__(self) -> None:
        self._history: deque[list[RGB]] = deque(maxlen=TEMPORAL_WINDOW)
        self._last_valid: Optional[list[RGB]] = None

    def detect(self, frame: np.ndarray, mode: str = "AUTO") -> tuple[Optional[list[RGB]], np.ndarray, float]:
        """
        Detect block colors in one frame.

        Returns (colors, debug_frame, confidence).
        - colors: list[(R,G,B)] or None
        - confidence: 0..1
        """
        debug = frame.copy()
        mode_normalized = (mode or "AUTO").upper()

        if mode_normalized == "FALLBACK":
            colors, debug, confidence = self._detect_center_strip(frame, debug)
        else:
            colors, debug, confidence = self._detect_with_screen_find(frame, debug)
            if colors is None or confidence < 0.40:
                colors, debug, confidence = self._detect_center_strip(frame, debug)

        if colors is not None:
            self._history.append(colors)
            colors = self._temporal_smooth()
            self._last_valid = colors
        else:
            colors = self._last_valid

        return colors, debug, confidence

    def reset_calibration(self) -> None:
        self._history.clear()
        self._last_valid = None

    def _detect_with_screen_find(self, frame: np.ndarray, debug: np.ndarray) -> tuple[Optional[list[RGB]], np.ndarray, float]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blurred, 30, 120)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, debug, 0.0

        best_quad: Optional[np.ndarray] = None
        best_area = 0.0
        frame_area = float(frame.shape[0] * frame.shape[1])

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_CONTOUR_AREA:
                continue

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) != 4:
                continue

            rect = cv2.minAreaRect(approx)
            w, h = rect[1]
            if w <= 1 or h <= 1:
                continue

            ratio = max(w, h) / min(w, h)
            if ratio > 6.0:
                continue

            if area > best_area:
                best_area = area
                best_quad = approx

        if best_quad is None:
            return None, debug, 0.0

        points = self._order_points(best_quad.reshape(4, 2).astype(np.float32))
        warped = self._perspective_warp(frame, points, width=960, height=160)
        colors = self._extract_blocks_from_strip(warped)

        cv2.drawContours(debug, [best_quad], -1, (0, 255, 100), 3)
        cv2.putText(debug, "Screen detect", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2)

        area_ratio = min(best_area / max(frame_area, 1.0), 0.30) / 0.30
        confidence = 0.70 + (0.20 * area_ratio)
        return colors, debug, float(min(confidence, 0.90))

    def _detect_center_strip(self, frame: np.ndarray, debug: np.ndarray) -> tuple[Optional[list[RGB]], np.ndarray, float]:
        h, w = frame.shape[:2]
        x1 = int(w * 0.05)
        x2 = int(w * 0.95)
        y1 = int(h * 0.35)
        y2 = int(h * 0.65)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None, debug, 0.0

        cv2.rectangle(debug, (x1, y1), (x2, y2), (255, 165, 0), 2)
        cv2.putText(debug, "Fallback strip", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

        colors = self._extract_blocks_from_strip(roi)
        return colors, debug, 0.55

    def _extract_blocks_from_strip(self, strip: np.ndarray) -> Optional[list[RGB]]:
        h, w = strip.shape[:2]
        if h <= 0 or w <= NUM_BLOCKS:
            return None

        block_w = w / float(NUM_BLOCKS)
        colors: list[RGB] = []

        for i in range(NUM_BLOCKS):
            bx1 = int(i * block_w)
            bx2 = int((i + 1) * block_w)
            block = strip[:, bx1:bx2]
            if block.size == 0:
                return None

            margin_x = max(int(block.shape[1] * (1.0 - BLOCK_SAMPLE_RATIO) / 2.0), 1)
            margin_y = max(int(block.shape[0] * (1.0 - BLOCK_SAMPLE_RATIO) / 2.0), 1)
            center = block[margin_y:-margin_y, margin_x:-margin_x]
            if center.size == 0:
                center = block

            center_norm = self._normalize_lighting(center)
            avg_bgr = cv2.mean(center_norm)[:3]
            colors.append((int(avg_bgr[2]), int(avg_bgr[1]), int(avg_bgr[0])))

        return colors

    def _normalize_lighting(self, patch: np.ndarray) -> np.ndarray:
        if patch.shape[0] < 4 or patch.shape[1] < 4:
            return patch
        try:
            lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
            l_chan, a_chan, b_chan = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            l_chan = clahe.apply(l_chan)
            normalized = cv2.merge((l_chan, a_chan, b_chan))
            return cv2.cvtColor(normalized, cv2.COLOR_LAB2BGR)
        except Exception:
            return patch

    def _temporal_smooth(self) -> list[RGB]:
        if not self._history:
            return [(255, 255, 255)] * NUM_BLOCKS

        smoothed: list[RGB] = []
        for block_idx in range(NUM_BLOCKS):
            rs = [frame[block_idx][0] for frame in self._history]
            gs = [frame[block_idx][1] for frame in self._history]
            bs = [frame[block_idx][2] for frame in self._history]
            smoothed.append((int(np.median(rs)), int(np.median(gs)), int(np.median(bs))))
        return smoothed

    @staticmethod
    def _order_points(points: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype=np.float32)
        sums = points.sum(axis=1)
        diffs = np.diff(points, axis=1)
        rect[0] = points[np.argmin(sums)]
        rect[2] = points[np.argmax(sums)]
        rect[1] = points[np.argmin(diffs)]
        rect[3] = points[np.argmax(diffs)]
        return rect

    @staticmethod
    def _perspective_warp(frame: np.ndarray, points: np.ndarray, width: int, height: int) -> np.ndarray:
        dst = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(points, dst)
        return cv2.warpPerspective(frame, matrix, (width, height))


def draw_block_overlay(frame: np.ndarray, block_colors: Optional[Sequence[RGB]], x: int = 10, y: Optional[int] = None, block_size: int = 40) -> np.ndarray:
    """Draw detected block colors as mini-squares on debug frame."""
    if block_colors is None:
        return frame

    h, w = frame.shape[:2]
    if y is None:
        y = h - block_size - 15

    total_w = NUM_BLOCKS * block_size + (NUM_BLOCKS - 1) * 4
    x = (w - total_w) // 2

    for i, (r, g, b) in enumerate(block_colors):
        bx = x + i * (block_size + 4)
        cv2.rectangle(frame, (bx, y), (bx + block_size, y + block_size), (b, g, r), -1)
        cv2.rectangle(frame, (bx, y), (bx + block_size, y + block_size), (200, 200, 200), 1)

    return frame
