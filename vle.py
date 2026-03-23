#!/usr/bin/env python3
"""
STATE-BASED VIDEO DATASET TOOL (CORRECT FINAL VERSION)
- State controls recording
- Clips auto-split at 10s
- State NEVER changes automatically
- No discard
- Old-tool behavior restored
"""

import cv2
import time
from pathlib import Path

# ================= CONFIG =================
INPUT_VIDEO = r"C:\Users\Rapportsoft\Documents\kalmarcam1\RTSP Server23.mp4"
OUTPUT_DIR = r"E:\kalmardatavideoscam302"

LABEL_KEYS = {
    'n': 'normal',
    'p': 'picked',
    'l': 'placed'
}

FORCED_FPS = 25
MAX_CLIP_SEC = 10
JUMP_STEP = 5
# =========================================

# Arrow keys (Windows)
LEFT  = 2424832
RIGHT = 2555904
UP    = 2490368
DOWN  = 2621440


# ================= RECORDER =================
class ClipRecorder:
    def __init__(self, root, fps, size):
        self.root = root
        self.fps = fps
        self.size = size

        self.writer = None
        self.label = None
        self.frames_written = 0
        self.current_path = None

        self.max_frames = int(MAX_CLIP_SEC * fps)

    def _open_new_clip(self):
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.current_path = self.root / self.label / f"{self.label}_{ts}.mp4"

        self.writer = cv2.VideoWriter(
            str(self.current_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            self.size
        )

        self.frames_written = 0
        print(f"[NEW CLIP] {self.current_path}")

    def start(self, label):
        self.stop()
        self.label = label
        self._open_new_clip()

    def write(self, frame):
        if not self.writer:
            return

        self.writer.write(frame)
        self.frames_written += 1

        # 🔥 AUTO-SPLIT (STATE PRESERVED)
        if self.frames_written >= self.max_frames:
            self.writer.release()
            print("[AUTO SPLIT] 10s reached")
            self._open_new_clip()

    def stop(self):
        if self.writer:
            self.writer.release()
            print("[STATE END]")
        self.writer = None
        self.label = None
        self.frames_written = 0
        self.current_path = None


# ================= MAIN =================
def run():
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise RuntimeError("❌ Cannot open video")

    fps = FORCED_FPS
    delay = int(1000 / fps)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_name = Path(INPUT_VIDEO).stem
    root = Path(OUTPUT_DIR) / video_name
    for lbl in LABEL_KEYS.values():
        (root / lbl).mkdir(parents=True, exist_ok=True)

    recorder = ClipRecorder(root, fps, (w, h))
    paused = False

    cv2.namedWindow("Kalmar Dataset Tool (STATE + SPLIT)", cv2.WINDOW_NORMAL)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            key = cv2.waitKeyEx(0)
            ret, frame = cap.read()
            if not ret:
                break

        # ---- Overlay ----
        view = frame.copy()
        if recorder.label:
            dur = recorder.frames_written / fps
            text = f"REC: {recorder.label} | {dur:.1f}s"
            color = (0, 255, 0)
        else:
            text = "IDLE"
            color = (255, 255, 0)

        cv2.putText(view, text, (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("Kalmar Dataset Tool (STATE + SPLIT)", view)

        key = cv2.waitKeyEx(0 if paused else delay)

        # ---- Navigation ----
        if paused:
            pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if key == LEFT:
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(pos - 1, 0))
                continue
            elif key == UP:
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos + JUMP_STEP)
                continue
            elif key == DOWN:
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(pos - JUMP_STEP, 0))
                continue

        ch = key & 0xFF

        # ---- Controls ----
        if ch == ord(' '):
            paused = not paused

        elif chr(ch) in LABEL_KEYS:
            recorder.start(LABEL_KEYS[chr(ch)])

        elif ch == ord('e'):
            recorder.stop()

        elif ch == ord('q'):
            recorder.stop()
            break

        # ---- Save ----
        if recorder.label and not paused:
            recorder.write(frame)

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Finished")


# ================= RUN =================
if __name__ == "__main__":
    run()
