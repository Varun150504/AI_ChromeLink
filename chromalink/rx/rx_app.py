"""
AI ChromaLink RX application.

Runs on the receiver laptop. Uses camera + CV detection pipeline to decode
8-block optical frames from the TX laptop and mirror the colors/message.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

# Allow local package imports when running this file directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.detector import ColorDetector, draw_block_overlay
from core.encoder import NUM_BLOCKS, WHITE, decode_blocks, is_idle, is_sync_end, is_sync_start

BG_DARK = "#0A0E1A"
BG_CARD = "#1C2333"
ACCENT_BLUE = "#3B82F6"
ACCENT_CYAN = "#06B6D4"
ACCENT_GREEN = "#10B981"
ACCENT_RED = "#EF4444"
ACCENT_AMBER = "#F59E0B"
ACCENT_PURPLE = "#8B5CF6"
TEXT_PRIMARY = "#F9FAFB"
TEXT_MUTED = "#6B7280"

DETECTION_FPS = 15
MIN_CONFIDENCE = 0.45


class RXApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AI ChromaLink - RX Receiver")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(True, True)

        # Runtime state.
        self._cap = None
        self._detector = ColorDetector()
        self._running = False

        self._receiving = False
        self._char_gate_open = False
        self._last_signature = None

        self._current_blocks = [WHITE] * NUM_BLOCKS
        self._decoded_message = []
        self._last_char = ""
        self._confidence = 0.0

        self._frames_total = 0
        self._frames_detected = 0
        self._chars_received = 0

        self._build_ui()
        self._set_window_size()
        self._update_status("OFFLINE", ACCENT_RED)
        self._list_cameras()

    def _set_window_size(self) -> None:
        width, height = 1200, 820
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = (sw - width) // 2
        y = (sh - height) // 2
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.minsize(1000, 700)

    def _build_ui(self) -> None:
        header = tk.Frame(self.root, bg="#060A14", height=64)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)

        tk.Label(
            header,
            text="AI ChromaLink",
            bg="#060A14",
            fg=ACCENT_CYAN,
            font=("Courier", 18, "bold"),
        ).pack(side=tk.LEFT, padx=24, pady=16)

        tk.Label(
            header,
            text="RX RECEIVER",
            bg="#060A14",
            fg=ACCENT_PURPLE,
            font=("Courier", 12, "bold"),
        ).pack(side=tk.LEFT, padx=8)

        status_frame = tk.Frame(header, bg="#060A14")
        status_frame.pack(side=tk.RIGHT, padx=24)

        tk.Label(
            status_frame,
            text="AI confidence:",
            bg="#060A14",
            fg=TEXT_MUTED,
            font=("Courier", 9),
        ).pack(side=tk.LEFT)

        self._conf_label = tk.Label(
            status_frame,
            text="0%",
            bg="#060A14",
            fg=ACCENT_GREEN,
            font=("Courier", 12, "bold"),
        )
        self._conf_label.pack(side=tk.LEFT, padx=6)

        self._status_dot = tk.Label(
            status_frame,
            text="●",
            bg="#060A14",
            fg=ACCENT_RED,
            font=("Courier", 14),
        )
        self._status_dot.pack(side=tk.LEFT, padx=(10, 2))

        self._status_label = tk.Label(
            status_frame,
            text="OFFLINE",
            bg="#060A14",
            fg=ACCENT_RED,
            font=("Courier", 11, "bold"),
        )
        self._status_label.pack(side=tk.LEFT)

        content = tk.Frame(self.root, bg=BG_DARK)
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=(10, 20))

        left = tk.Frame(content, bg=BG_DARK)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        right = tk.Frame(content, bg=BG_DARK, width=330)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right.pack_propagate(False)

        self._build_camera_panel(left)
        self._build_rx_display(left)
        self._build_message_panel(left)
        self._build_controls_panel(right)
        self._build_log_panel(right)

    def _build_camera_panel(self, parent: tk.Widget) -> None:
        card = self._make_card(parent, "LIVE CAMERA FEED")

        self._camera_label = tk.Label(
            card,
            bg=BG_DARK,
            text="Camera offline\nPress START DETECTION",
            fg=TEXT_MUTED,
            font=("Courier", 11),
            height=14,
        )
        self._camera_label.pack(fill=tk.BOTH, expand=True)

    def _build_rx_display(self, parent: tk.Widget) -> None:
        card = self._make_card(parent, "REPLICATED 8-BLOCK DISPLAY")

        self._rx_canvas = tk.Canvas(card, bg=BG_CARD, height=120, highlightthickness=0)
        self._rx_canvas.pack(fill=tk.X, pady=(0, 6))
        self._rx_canvas.bind("<Configure>", lambda _: self._refresh_rx_blocks())

        labels = tk.Frame(card, bg=BG_CARD)
        labels.pack(fill=tk.X)
        for i in range(NUM_BLOCKS):
            tk.Label(labels, text=f"B{i}", bg=BG_CARD, fg=TEXT_MUTED, font=("Courier", 8)).pack(
                side=tk.LEFT, expand=True
            )

        self._rx_info = tk.Label(
            card,
            text="Waiting for TX signal...",
            bg=BG_CARD,
            fg=TEXT_MUTED,
            font=("Courier", 9),
        )
        self._rx_info.pack(pady=(8, 0))

    def _build_message_panel(self, parent: tk.Widget) -> None:
        card = self._make_card(parent, "DECODED MESSAGE")

        msg_outer = tk.Frame(card, bg=ACCENT_PURPLE, padx=2, pady=2)
        msg_outer.pack(fill=tk.X, pady=(0, 10))

        self._msg_display = tk.Label(
            msg_outer,
            text="",
            bg=BG_DARK,
            fg=ACCENT_GREEN,
            font=("Courier", 20, "bold"),
            height=2,
            anchor=tk.W,
            padx=12,
        )
        self._msg_display.pack(fill=tk.X)

        row = tk.Frame(card, bg=BG_CARD)
        row.pack(fill=tk.X)

        tk.Label(row, text="Last char:", bg=BG_CARD, fg=TEXT_MUTED, font=("Courier", 9)).pack(side=tk.LEFT)
        self._last_char_label = tk.Label(
            row,
            text="-",
            bg=BG_CARD,
            fg=ACCENT_CYAN,
            font=("Courier", 14, "bold"),
        )
        self._last_char_label.pack(side=tk.LEFT, padx=8)

        tk.Label(row, text="ASCII:", bg=BG_CARD, fg=TEXT_MUTED, font=("Courier", 9)).pack(side=tk.LEFT, padx=(20, 0))
        self._ascii_label = tk.Label(
            row,
            text="-",
            bg=BG_CARD,
            fg=ACCENT_AMBER,
            font=("Courier", 14, "bold"),
        )
        self._ascii_label.pack(side=tk.LEFT, padx=8)

        self._make_button(card, "CLEAR MESSAGE", ACCENT_RED, self._clear_message).pack(fill=tk.X, pady=(10, 0))

    def _build_controls_panel(self, parent: tk.Widget) -> None:
        card = self._make_card(parent, "CAMERA CONTROLS")
        card.pack_configure(fill=tk.X)

        tk.Label(card, text="Camera source:", bg=BG_CARD, fg=TEXT_MUTED, font=("Courier", 9)).pack(anchor=tk.W)

        self._cam_var = tk.IntVar(value=0)
        cam_frame = tk.Frame(card, bg=BG_CARD)
        cam_frame.pack(fill=tk.X, pady=(4, 10))
        for index in range(3):
            tk.Radiobutton(
                cam_frame,
                text=f"Cam {index}",
                variable=self._cam_var,
                value=index,
                bg=BG_CARD,
                fg=TEXT_PRIMARY,
                selectcolor=BG_DARK,
                activebackground=BG_CARD,
                font=("Courier", 9),
            ).pack(side=tk.LEFT, padx=4)

        self._cam_status = tk.Label(card, text="Checking cameras...", bg=BG_CARD, fg=TEXT_MUTED, font=("Courier", 8))
        self._cam_status.pack(anchor=tk.W, pady=(0, 8))

        self._make_button(card, "START DETECTION", ACCENT_GREEN, self._start_camera).pack(fill=tk.X, pady=(0, 6))
        self._make_button(card, "STOP", ACCENT_RED, self._stop_camera).pack(fill=tk.X, pady=(0, 6))
        self._make_button(card, "RESET AI CALIBRATION", ACCENT_BLUE, self._reset_calibration).pack(fill=tk.X)

        tk.Label(card, text="\nDetection mode:", bg=BG_CARD, fg=TEXT_MUTED, font=("Courier", 9)).pack(anchor=tk.W)
        self._mode_var = tk.StringVar(value="AUTO")

        tk.Radiobutton(
            card,
            text="AUTO (screen contour + fallback)",
            variable=self._mode_var,
            value="AUTO",
            bg=BG_CARD,
            fg=TEXT_PRIMARY,
            selectcolor=BG_DARK,
            activebackground=BG_CARD,
            font=("Courier", 8),
        ).pack(anchor=tk.W)

        tk.Radiobutton(
            card,
            text="FALLBACK (center strip only)",
            variable=self._mode_var,
            value="FALLBACK",
            bg=BG_CARD,
            fg=TEXT_PRIMARY,
            selectcolor=BG_DARK,
            activebackground=BG_CARD,
            font=("Courier", 8),
        ).pack(anchor=tk.W)

        tk.Label(card, text="\nDetection stats:", bg=BG_CARD, fg=TEXT_MUTED, font=("Courier", 9)).pack(anchor=tk.W)
        self._stats_label = tk.Label(
            card,
            text="Frames: 0\nDetected: 0\nChars RX: 0",
            bg=BG_CARD,
            fg=ACCENT_CYAN,
            font=("Courier", 8),
            justify=tk.LEFT,
        )
        self._stats_label.pack(anchor=tk.W)

    def _build_log_panel(self, parent: tk.Widget) -> None:
        card = self._make_card(parent, "RX LOG")
        card.pack_configure(fill=tk.BOTH, expand=True)

        self._log_text = tk.Text(
            card,
            bg=BG_DARK,
            fg=ACCENT_PURPLE,
            font=("Courier", 8),
            relief=tk.FLAT,
            state=tk.DISABLED,
            wrap=tk.WORD,
        )
        self._log_text.pack(fill=tk.BOTH, expand=True)

        self._log("AI ChromaLink RX initialized")
        self._log("Awaiting camera start...")
        self._log("-" * 30)

    def _start_camera(self) -> None:
        if self._running:
            return

        cam_idx = self._cam_var.get()
        self._cap = cv2.VideoCapture(cam_idx)
        if not self._cap.isOpened():
            self._log(f"ERROR: cannot open camera {cam_idx}")
            self._update_status("ERROR", ACCENT_RED)
            return

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self._cap.set(cv2.CAP_PROP_FPS, DETECTION_FPS)

        self._receiving = False
        self._char_gate_open = False
        self._last_signature = None

        self._running = True
        self._update_status("SCANNING", ACCENT_AMBER)
        self._log(f"Camera {cam_idx} started in {self._mode_var.get()} mode")

        threading.Thread(target=self._camera_loop, daemon=True).start()

    def _stop_camera(self) -> None:
        self._running = False
        time.sleep(0.15)
        if self._cap is not None:
            self._cap.release()
            self._cap = None

        self._update_status("OFFLINE", ACCENT_RED)
        self._camera_label.configure(image="", text="Camera stopped", fg=TEXT_MUTED)
        self._log("Camera stopped")

    def _camera_loop(self) -> None:
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                break

            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            self._frames_total += 1

            detect_mode = self._mode_var.get()
            colors, debug_frame, confidence = self._detector.detect(frame, mode=detect_mode)
            self._confidence = confidence

            if colors is not None:
                self._frames_detected += 1
                self._process_detected_colors(colors, confidence)
                debug_frame = draw_block_overlay(debug_frame, colors)

            debug_frame = self._draw_hud(debug_frame, confidence, detect_mode)

            self.root.after(0, self._update_camera_view, debug_frame)
            self.root.after(0, self._update_stats)

            time.sleep(1.0 / DETECTION_FPS)

    def _process_detected_colors(self, colors, confidence: float) -> None:
        # Mirror block colors on RX display even for low confidence frames.
        self._current_blocks = colors
        self.root.after(0, self._refresh_rx_blocks)

        if confidence < MIN_CONFIDENCE:
            self.root.after(0, self._update_rx_info, confidence)
            return

        if is_sync_start(colors):
            if not self._receiving:
                self._receiving = True
                self._decoded_message = []
                self._chars_received = 0
                self._last_signature = None
                self.root.after(0, self._update_status, "RECEIVING", ACCENT_GREEN)
                self.root.after(0, self._log, "SYNC START detected - receiving")
            self._char_gate_open = True
            self.root.after(0, self._update_rx_info, confidence)
            return

        if is_sync_end(colors):
            if self._receiving:
                self._receiving = False
                self._char_gate_open = False
                self._last_signature = None
                message = "".join(self._decoded_message)
                self.root.after(0, self._update_status, "SCANNING", ACCENT_AMBER)
                self.root.after(0, self._log, f"SYNC END - message: '{message}'")
            self.root.after(0, self._update_rx_info, confidence)
            return

        if is_idle(colors):
            # GAP frame between characters.
            if self._receiving:
                self._char_gate_open = True
                self._last_signature = None
            self.root.after(0, self._update_rx_info, confidence)
            return

        char = decode_blocks(colors)

        if not self._receiving:
            self.root.after(0, self._update_rx_info, confidence)
            return

        if not char:
            self._char_gate_open = True
            self._last_signature = None
            self.root.after(0, self._update_rx_info, confidence)
            return

        signature = self._frame_signature(colors)
        should_accept = self._char_gate_open or (signature != self._last_signature)

        if should_accept:
            self._decoded_message.append(char)
            self._chars_received += 1
            self._last_char = char
            self._char_gate_open = False
            self._last_signature = signature
            self.root.after(0, self._on_char_received, char)

        self.root.after(0, self._update_rx_info, confidence)

    @staticmethod
    def _frame_signature(colors):
        # White block = 0, non-white block = 1.
        return tuple(1 if not (r > 220 and g > 220 and b > 220) else 0 for r, g, b in colors)

    def _on_char_received(self, char: str) -> None:
        message = "".join(self._decoded_message)
        self._msg_display.configure(text=message)
        self._last_char_label.configure(text=repr(char))
        self._ascii_label.configure(text=str(ord(char)))
        self._log(f"RX '{char}' ASCII={ord(char)}")

    def _update_rx_info(self, confidence: float) -> None:
        self._rx_info.configure(
            text=(
                f"Confidence: {confidence:.0%}  |  "
                f"Receiving: {'YES' if self._receiving else 'NO'}  |  "
                f"Chars: {self._chars_received}"
            )
        )

    def _refresh_rx_blocks(self) -> None:
        canvas = self._rx_canvas
        canvas.delete("all")

        width = canvas.winfo_width()
        height = canvas.winfo_height() or 120
        if width < 8:
            return

        gap = 4
        block_w = (width - gap * (NUM_BLOCKS - 1) - 8) / NUM_BLOCKS

        for i, (r, g, b) in enumerate(self._current_blocks):
            x1 = 4 + i * (block_w + gap)
            x2 = x1 + block_w
            y1, y2 = 8, height - 8
            color = f"#{r:02X}{g:02X}{b:02X}"

            canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="#2D3748", width=2)

            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = "#000000" if luminance > 128 else "#CCCCCC"
            canvas.create_text(x1 + block_w / 2, y1 + 16, text=f"{r}", fill=text_color, font=("Courier", 7))
            canvas.create_text(x1 + block_w / 2, y1 + 28, text=f"{g}", fill=text_color, font=("Courier", 7))
            canvas.create_text(x1 + block_w / 2, y1 + 40, text=f"{b}", fill=text_color, font=("Courier", 7))

    def _update_camera_view(self, frame) -> None:
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            label_w = self._camera_label.winfo_width() or 700
            label_h = self._camera_label.winfo_height() or 350
            h, w = frame_rgb.shape[:2]
            scale = min(label_w / w, label_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv2.resize(frame_rgb, (new_w, new_h))
            image = ImageTk.PhotoImage(Image.fromarray(resized))
            self._camera_label.configure(image=image, text="")
            self._camera_label._img = image
        except Exception:
            pass

    def _draw_hud(self, frame, confidence: float, mode: str):
        h, w = frame.shape[:2]

        bar_w = int(w * 0.30)
        bar_h = 12
        bx, by = 10, h - 30

        cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (40, 40, 40), -1)
        fill = int(bar_w * max(0.0, min(confidence, 1.0)))
        if confidence > 0.70:
            color = (20, 200, 80)
        elif confidence > 0.45:
            color = (200, 160, 20)
        else:
            color = (200, 40, 40)
        cv2.rectangle(frame, (bx, by), (bx + fill, by + bar_h), color, -1)

        cv2.putText(
            frame,
            f"AI Confidence: {confidence:.0%} | Mode: {mode}",
            (bx, by - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        if self._receiving:
            cv2.circle(frame, (w - 30, 30), 12, (20, 220, 80), -1)
            cv2.putText(frame, "RX", (w - 56, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 220, 80), 1)

        cv2.putText(frame, f"Frame #{self._frames_total}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        return frame

    def _update_stats(self) -> None:
        rate = (self._frames_detected / self._frames_total * 100.0) if self._frames_total else 0.0
        self._stats_label.configure(
            text=f"Frames:   {self._frames_total}\n"
            f"Detected: {self._frames_detected} ({rate:.0f}%)\n"
            f"Chars RX: {self._chars_received}"
        )
        self._conf_label.configure(text=f"{self._confidence:.0%}")

    def _clear_message(self) -> None:
        self._decoded_message = []
        self._chars_received = 0
        self._last_char = ""
        self._msg_display.configure(text="")
        self._last_char_label.configure(text="-")
        self._ascii_label.configure(text="-")
        self._log("Message cleared")

    def _reset_calibration(self) -> None:
        self._detector.reset_calibration()
        self._log("AI calibration reset")

    def _list_cameras(self) -> None:
        available = []
        for index in range(3):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                available.append(index)
                cap.release()

        if available:
            self._cam_status.configure(text=f"Available: {available}", fg=ACCENT_GREEN)
        else:
            self._cam_status.configure(text="No cameras found", fg=ACCENT_RED)

    def _update_status(self, text: str, color: str) -> None:
        self._status_dot.configure(fg=color)
        self._status_label.configure(text=text, fg=color)

    def _make_card(self, parent: tk.Widget, title: str) -> tk.Frame:
        outer = tk.Frame(parent, bg=ACCENT_PURPLE, padx=1, pady=1)
        outer.pack(fill=tk.X, pady=6)

        card = tk.Frame(outer, bg=BG_CARD, padx=16, pady=12)
        card.pack(fill=tk.BOTH, expand=True)

        tk.Label(card, text=title, bg=BG_CARD, fg=ACCENT_PURPLE, font=("Courier", 9, "bold")).pack(
            anchor=tk.W, pady=(0, 8)
        )
        return card

    def _make_button(self, parent: tk.Widget, text: str, color: str, command) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=color,
            fg=BG_DARK,
            font=("Courier", 9, "bold"),
            relief=tk.FLAT,
            cursor="hand2",
            padx=10,
            pady=8,
            activebackground=color,
            activeforeground=BG_DARK,
        )

    def _log(self, message: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self._log_text.configure(state=tk.NORMAL)
        self._log_text.insert(tk.END, f"[{ts}] {message}\n")
        self._log_text.see(tk.END)
        self._log_text.configure(state=tk.DISABLED)

    def on_close(self) -> None:
        self._stop_camera()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    style = ttk.Style()
    style.theme_use("clam")

    app = RXApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
