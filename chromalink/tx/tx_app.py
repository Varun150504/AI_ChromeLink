"""
AI ChromaLink — TX (Transmitter) Application
=============================================
Professional GUI dashboard for the transmitter side.
Encodes text messages into 8-block RGB color patterns and displays them
for the RX system to detect via camera.

Run: python tx_app.py
"""

import tkinter as tk
from tkinter import ttk, messagebox, font as tkfont
import sys
import os
import time
import threading

# Allow imports from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.encoder import (
    encode_character, encode_message, get_idle_frame,
    get_sync_start_frame, get_sync_end_frame,
    WHITE, SYNC_START, SYNC_END, IDLE_COLOR, NUM_BLOCKS
)

# ── Design tokens ─────────────────────────────────────────────────────────────
BG_DARK      = "#0A0E1A"
BG_PANEL     = "#111827"
BG_CARD      = "#1C2333"
ACCENT_BLUE  = "#3B82F6"
ACCENT_CYAN  = "#06B6D4"
ACCENT_GREEN = "#10B981"
ACCENT_RED   = "#EF4444"
ACCENT_AMBER = "#F59E0B"
TEXT_PRIMARY = "#F9FAFB"
TEXT_MUTED   = "#6B7280"
TEXT_DIM     = "#374151"
BORDER       = "#1F2937"

# Transmission timing
FRAME_DURATION = 0.8    # seconds per character frame
SYNC_DURATION = 0.6     # seconds for sync frames
GAP_DURATION_FACTOR = 0.35  # fraction of char duration for inter-char gap


class TXApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI ChromaLink — TX Transmitter")
        self.root.configure(bg=BG_DARK)
        self.root.resizable(True, True)

        # State
        self._current_blocks  = get_idle_frame()
        self._transmitting    = False
        self._tx_thread       = None
        self._message_queue   = []
        self._tx_log          = []
        self._char_index      = 0
        self._total_chars     = 0
        self._manual_colors   = [WHITE] * NUM_BLOCKS

        self._build_ui()
        self._set_window_size()
        self._refresh_blocks()
        self._update_status("IDLE", ACCENT_AMBER)

    # ── Window setup ──────────────────────────────────────────────────────────

    def _set_window_size(self):
        w, h = 1400, 1000
        sw   = self.root.winfo_screenwidth()
        sh   = self.root.winfo_screenheight()
        x    = (sw - w) // 2
        y    = (sh - h) // 2
        self.root.geometry(f"{w}x{h}+{x}+{y}")
        self.root.minsize(1200, 800)

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Top header bar ────────────────────────────────────────────────────
        header = tk.Frame(self.root, bg="#060A14", height=64)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)

        tk.Label(header, text="⬡  AI ChromaLink",
                 bg="#060A14", fg=ACCENT_CYAN,
                 font=("Courier", 18, "bold")).pack(side=tk.LEFT, padx=24, pady=16)

        tk.Label(header, text="TX  TRANSMITTER",
                 bg="#060A14", fg=ACCENT_BLUE,
                 font=("Courier", 12, "bold")).pack(side=tk.LEFT, padx=8)

        # Status indicator (top right)
        self._status_frame = tk.Frame(header, bg="#060A14")
        self._status_frame.pack(side=tk.RIGHT, padx=24)
        self._status_dot = tk.Label(self._status_frame, text="●",
                                     bg="#060A14", fg=ACCENT_AMBER,
                                     font=("Courier", 14))
        self._status_dot.pack(side=tk.LEFT)
        self._status_label = tk.Label(self._status_frame, text="IDLE",
                                       bg="#060A14", fg=ACCENT_AMBER,
                                       font=("Courier", 11, "bold"))
        self._status_label.pack(side=tk.LEFT, padx=6)

        # ── Main content ──────────────────────────────────────────────────────
        content = tk.Frame(self.root, bg=BG_DARK)
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=(10, 20))

        # Left column — controls
        left = tk.Frame(content, bg=BG_DARK)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Right column — log + info
        right = tk.Frame(content, bg=BG_DARK, width=300)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right.pack_propagate(False)

        self._build_display_panel(left)
        self._build_message_panel(left)
        self._build_manual_panel(left)
        self._build_log_panel(right)
        self._build_info_panel(right)

    def _build_display_panel(self, parent):
        """The big 8-block color display — the core visual element."""
        card = self._make_card(parent, "OPTICAL TRANSMISSION DISPLAY")

        # Block display canvas
        canvas_frame = tk.Frame(card, bg=BG_CARD)
        canvas_frame.pack(fill=tk.X, pady=(0, 12))

        self._block_canvas = tk.Canvas(canvas_frame, bg=BG_CARD,
                                        height=300, highlightthickness=0)
        self._block_canvas.pack(fill=tk.X, padx=4)
        self._block_canvas.bind("<Configure>", lambda e: self._refresh_blocks())

        # Block labels
        label_frame = tk.Frame(card, bg=BG_CARD)
        label_frame.pack(fill=tk.X)
        for i in range(NUM_BLOCKS):
            lf = tk.Frame(label_frame, bg=BG_CARD)
            lf.pack(side=tk.LEFT, expand=True)
            tk.Label(lf, text=f"B{i}", bg=BG_CARD, fg=TEXT_MUTED,
                     font=("Courier", 8)).pack()

        # Current frame info
        self._frame_info = tk.Label(card, text="Frame: IDLE  |  Char: —  |  Code: —",
                                     bg=BG_CARD, fg=TEXT_MUTED,
                                     font=("Courier", 9))
        self._frame_info.pack(pady=(10, 0))

        # Progress bar
        prog_frame = tk.Frame(card, bg=BG_CARD)
        prog_frame.pack(fill=tk.X, pady=(8, 0))
        tk.Label(prog_frame, text="TX Progress", bg=BG_CARD, fg=TEXT_MUTED,
                 font=("Courier", 8)).pack(anchor=tk.W)
        self._progress = ttk.Progressbar(prog_frame, mode='determinate',
                                          maximum=100, value=0)
        self._progress.pack(fill=tk.X, pady=2)
        self._progress_label = tk.Label(prog_frame, text="0 / 0 characters",
                                         bg=BG_CARD, fg=TEXT_MUTED,
                                         font=("Courier", 8))
        self._progress_label.pack(anchor=tk.W)

    def _build_message_panel(self, parent):
        """Message input and send controls."""
        card = self._make_card(parent, "MESSAGE ENCODER")

        # Message input
        tk.Label(card, text="Enter message to transmit:",
                 bg=BG_CARD, fg=TEXT_MUTED, font=("Courier", 9)).pack(anchor=tk.W)

        self._msg_var = tk.StringVar()
        entry_frame = tk.Frame(card, bg=ACCENT_BLUE, padx=2, pady=2)
        entry_frame.pack(fill=tk.X, pady=(4, 10))
        self._msg_entry = tk.Entry(entry_frame,
                                    textvariable=self._msg_var,
                                    bg=BG_DARK, fg=TEXT_PRIMARY,
                                    insertbackground=ACCENT_CYAN,
                                    font=("Courier", 13),
                                    relief=tk.FLAT, bd=6)
        self._msg_entry.pack(fill=tk.X)
        self._msg_entry.bind("<Return>", lambda e: self._send_message())

        # Speed control
        speed_frame = tk.Frame(card, bg=BG_CARD)
        speed_frame.pack(fill=tk.X, pady=(0, 10))
        tk.Label(speed_frame, text="TX Speed (s/char):",
                 bg=BG_CARD, fg=TEXT_MUTED, font=("Courier", 9)).pack(side=tk.LEFT)
        self._speed_var = tk.DoubleVar(value=FRAME_DURATION)
        speed_slider = ttk.Scale(speed_frame, from_=0.3, to=2.0,
                                  variable=self._speed_var, orient=tk.HORIZONTAL)
        speed_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        self._speed_val_label = tk.Label(speed_frame, text="0.80s",
                                          bg=BG_CARD, fg=ACCENT_CYAN,
                                          font=("Courier", 9))
        self._speed_val_label.pack(side=tk.LEFT)
        self._speed_var.trace('w', self._update_speed_label)

        # Buttons
        btn_frame = tk.Frame(card, bg=BG_CARD)
        btn_frame.pack(fill=tk.X)

        self._send_btn = self._make_button(
            btn_frame, "▶  TRANSMIT", ACCENT_GREEN, self._send_message)
        self._send_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 6))

        self._stop_btn = self._make_button(
            btn_frame, "■  STOP", ACCENT_RED, self._stop_transmission)
        self._stop_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)

        self._idle_btn = self._make_button(
            btn_frame, "◎  IDLE", TEXT_MUTED, self._set_idle)
        self._idle_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(6, 0))

    def _build_manual_panel(self, parent):
        """Manual block color assignment — for demo / custom patterns."""
        card = self._make_card(parent, "MANUAL BLOCK CONTROL  (for custom demo)")

        grid = tk.Frame(card, bg=BG_CARD)
        grid.pack(fill=tk.X)

        self._color_buttons = []
        self._color_labels  = []

        for i in range(NUM_BLOCKS):
            col_frame = tk.Frame(grid, bg=BG_CARD)
            col_frame.pack(side=tk.LEFT, expand=True, padx=2)

            tk.Label(col_frame, text=f"B{i}", bg=BG_CARD, fg=TEXT_MUTED,
                     font=("Courier", 8)).pack()

            preview = tk.Label(col_frame, bg=self._rgb_to_hex(WHITE),
                                width=4, height=2, relief=tk.FLAT,
                                cursor="hand2")
            preview.pack(pady=2)
            preview.bind("<Button-1>", lambda e, idx=i: self._pick_color(idx))
            self._color_labels.append(preview)

            rgb_lbl = tk.Label(col_frame, text="255\n255\n255",
                                bg=BG_CARD, fg=TEXT_DIM, font=("Courier", 6))
            rgb_lbl.pack()
            self._color_buttons.append(rgb_lbl)

        apply_btn = self._make_button(card, "⬡  DISPLAY MANUAL PATTERN",
                                       ACCENT_BLUE, self._apply_manual)
        apply_btn.pack(fill=tk.X, pady=(10, 0))

    def _build_log_panel(self, parent):
        """Transmission log."""
        card = self._make_card(parent, "TX LOG")
        card.pack_configure(fill=tk.BOTH, expand=True)

        self._log_text = tk.Text(card, bg=BG_DARK, fg=ACCENT_GREEN,
                                  font=("Courier", 8), relief=tk.FLAT,
                                  state=tk.DISABLED, wrap=tk.WORD,
                                  insertbackground=ACCENT_CYAN)
        self._log_text.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(card, command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=scrollbar.set)

        self._log("AI ChromaLink TX initialized")
        self._log("Ready to transmit")
        self._log("─" * 30)

    def _build_info_panel(self, parent):
        """Protocol info box."""
        card = self._make_card(parent, "PROTOCOL INFO")

        info_lines = [
            ("Blocks",    "8 optical bits"),
            ("Encoding",  "8-bit ASCII"),
            ("Colors",    "HSV hue mapping"),
            ("AI Layer",  "CV + temporal avg"),
            ("Sync",      "RED=START BLUE=END"),
            ("Gap",       "Dark frame between chars"),
            ("Idle",      "All dark blocks"),
        ]
        for label, val in info_lines:
            row = tk.Frame(card, bg=BG_CARD)
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=f"{label}:", bg=BG_CARD, fg=TEXT_MUTED,
                     font=("Courier", 8), width=10, anchor=tk.W).pack(side=tk.LEFT)
            tk.Label(row, text=val, bg=BG_CARD, fg=ACCENT_CYAN,
                     font=("Courier", 8), anchor=tk.W).pack(side=tk.LEFT)

    # ── Block rendering ───────────────────────────────────────────────────────

    def _refresh_blocks(self):
        """Redraw all 8 blocks on canvas."""
        canvas = self._block_canvas
        canvas.delete("all")

        W = canvas.winfo_width()
        H = canvas.winfo_height() or 140
        if W < 8:
            return

        gap       = 4
        total_gap = gap * (NUM_BLOCKS - 1)
        bw        = (W - total_gap - 8) / NUM_BLOCKS

        for i, (r, g, b) in enumerate(self._current_blocks):
            x1 = 4 + i * (bw + gap)
            x2 = x1 + bw
            y1 = 8
            y2 = H - 8
            hex_col = self._rgb_to_hex((r, g, b))

            # Block fill
            canvas.create_rectangle(x1, y1, x2, y2, fill=hex_col,
                                     outline="#2D3748", width=2)
            # Bit index label
            canvas.create_text(x1 + bw/2, y2 - 12,
                                text=f"B{i}", fill="#4B5563",
                                font=("Courier", 8))
            # RGB value
            canvas.create_text(x1 + bw/2, y1 + 20,
                                text=f"{r}", fill=self._contrast_color(r, g, b),
                                font=("Courier", 7))
            canvas.create_text(x1 + bw/2, y1 + 32,
                                text=f"{g}", fill=self._contrast_color(r, g, b),
                                font=("Courier", 7))
            canvas.create_text(x1 + bw/2, y1 + 44,
                                text=f"{b}", fill=self._contrast_color(r, g, b),
                                font=("Courier", 7))

    def _set_blocks(self, blocks, frame_label="", char="", code=""):
        """Update block display on main thread."""
        self._current_blocks = blocks
        self.root.after(0, self._refresh_blocks)
        info = f"Frame: {frame_label or 'CUSTOM'}  |  Char: {char or '—'}  |  Code: {code or '—'}"
        self.root.after(0, lambda: self._frame_info.configure(text=info))

    # ── Transmission logic ────────────────────────────────────────────────────

    def _send_message(self):
        msg = self._msg_var.get().strip()
        if not msg:
            messagebox.showwarning("Empty Message", "Please enter a message to transmit.")
            return
        if self._transmitting:
            messagebox.showinfo("Busy", "Already transmitting. Stop first.")
            return

        self._transmitting = True
        self._total_chars  = len(msg)
        self._char_index   = 0
        self._update_status("TRANSMITTING", ACCENT_GREEN)
        self._log(f"TX START: '{msg}'  ({len(msg)} chars)")

        self._tx_thread = threading.Thread(
            target=self._transmit_message, args=(msg,), daemon=True)
        self._tx_thread.start()

    def _transmit_message(self, msg: str):
        speed = self._speed_var.get()
        frames = encode_message(msg)

        for i, (label, blocks) in enumerate(frames):
            if not self._transmitting:
                break

            is_sync = label.startswith("SYNC")
            is_gap = label == "GAP"
            char = label.split(":")[-1] if "CHAR" in label else ""
            code = str(ord(char)) if char else ""

            self._set_blocks(blocks, label, char, code)

            if "CHAR" in label:
                self._char_index += 1
                self.root.after(0, self._update_progress)
                self.root.after(0, lambda c=char: self._log(
                    f"  [{self._char_index:3d}/{self._total_chars}] '{c}'  "
                    f"ASCII={ord(c)}  Hue={ord(c)/128:.3f}"))

            if is_sync:
                duration = SYNC_DURATION
            elif is_gap:
                duration = max(0.10, speed * GAP_DURATION_FACTOR)
            else:
                duration = speed
            time.sleep(duration)

        if self._transmitting:
            # Hold end sync briefly then go idle
            time.sleep(0.4)
            self._set_idle_internal()
            self.root.after(0, lambda: self._log(f"TX COMPLETE ✓"))
            self.root.after(0, lambda: self._update_status("IDLE", ACCENT_AMBER))

        self._transmitting = False

    def _stop_transmission(self):
        self._transmitting = False
        self._log("TX STOPPED by user")
        self._set_idle_internal()
        self._update_status("IDLE", ACCENT_AMBER)
        self._progress['value'] = 0
        self._progress_label.configure(text="0 / 0 characters")

    def _set_idle(self):
        if self._transmitting:
            self._stop_transmission()
        else:
            self._set_idle_internal()

    def _set_idle_internal(self):
        from core.encoder import get_idle_frame
        idle = get_idle_frame()
        self._set_blocks(idle, "IDLE")
        self.root.after(0, lambda: self._update_status("IDLE", ACCENT_AMBER))

    # ── Manual mode ───────────────────────────────────────────────────────────

    def _pick_color(self, idx):
        """Open color picker for a specific block."""
        from tkinter.colorchooser import askcolor
        current = self._manual_colors[idx]
        hex_cur = self._rgb_to_hex(current)
        result  = askcolor(color=hex_cur,
                           title=f"Block {idx} Color",
                           parent=self.root)
        if result[0]:
            r, g, b = int(result[0][0]), int(result[0][1]), int(result[0][2])
            self._manual_colors[idx] = (r, g, b)
            self._color_labels[idx].configure(bg=self._rgb_to_hex((r, g, b)))
            self._color_buttons[idx].configure(text=f"{r}\n{g}\n{b}")

    def _apply_manual(self):
        if self._transmitting:
            messagebox.showinfo("Busy", "Stop transmission first.")
            return
        self._set_blocks(self._manual_colors, "MANUAL")
        self._update_status("MANUAL", ACCENT_BLUE)
        self._log("Manual pattern applied")

    # ── UI Helpers ────────────────────────────────────────────────────────────

    def _make_card(self, parent, title):
        outer = tk.Frame(parent, bg=ACCENT_BLUE, padx=1, pady=1)
        outer.pack(fill=tk.X, pady=6)
        card = tk.Frame(outer, bg=BG_CARD, padx=16, pady=12)
        card.pack(fill=tk.BOTH, expand=True)
        tk.Label(card, text=title, bg=BG_CARD, fg=ACCENT_BLUE,
                 font=("Courier", 9, "bold")).pack(anchor=tk.W, pady=(0, 8))
        return card

    def _make_button(self, parent, text, color, command):
        btn = tk.Button(parent, text=text, command=command,
                        bg=color, fg=BG_DARK if color not in (TEXT_MUTED,) else BG_DARK,
                        font=("Courier", 9, "bold"),
                        relief=tk.FLAT, cursor="hand2",
                        padx=10, pady=8,
                        activebackground=color,
                        activeforeground=BG_DARK)
        return btn

    def _update_status(self, text, color):
        self._status_dot.configure(fg=color)
        self._status_label.configure(text=text, fg=color)

    def _update_progress(self):
        pct = (self._char_index / self._total_chars * 100) if self._total_chars else 0
        self._progress['value'] = pct
        self._progress_label.configure(
            text=f"{self._char_index} / {self._total_chars} characters")

    def _update_speed_label(self, *args):
        val = self._speed_var.get()
        self._speed_val_label.configure(text=f"{val:.2f}s")

    def _log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self._log_text.configure(state=tk.NORMAL)
        self._log_text.insert(tk.END, f"[{ts}] {msg}\n")
        self._log_text.see(tk.END)
        self._log_text.configure(state=tk.DISABLED)

    @staticmethod
    def _rgb_to_hex(rgb):
        r, g, b = rgb
        return f"#{r:02X}{g:02X}{b:02X}"

    @staticmethod
    def _contrast_color(r, g, b):
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return "#000000" if luminance > 128 else "#CCCCCC"


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()

    # Style
    style = ttk.Style()
    style.theme_use('clam')
    style.configure("TProgressbar",
                    troughcolor=BG_DARK,
                    background=ACCENT_GREEN,
                    bordercolor=BG_DARK,
                    lightcolor=ACCENT_GREEN,
                    darkcolor=ACCENT_GREEN)
    style.configure("TScale",
                    background=BG_CARD,
                    troughcolor=BG_DARK)

    app = TXApp(root)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()


if __name__ == "__main__":
    main()
