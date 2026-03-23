# kalmar_live_rtsp_fast.py
# ============================================================
# Fast Kalmar Action Detection — RTSP live + threaded capture
# Model predicts only "normal" and "picked". "placed" inferred by FSM.
# Optimized for low latency and faster preprocessing.
# ============================================================

import cv2
import torch
import numpy as np
import torchvision.models as models
import torch.nn as nn
from collections import deque
import time
import threading
import sys

# ---------------- Windows / LSTM safety ----------------
torch.backends.cudnn.enabled = False

# ===================== CONFIG =====================
# Set RTSP_URL for live phone stream (recommended) or set VIDEO_PATH to local file.
RTSP_URL = "rtsp://192.168.1.16:8080/h264_ulaw.sdp"   # <- change to your phone RTSP
VIDEO_PATH = None  # If set, this will be used instead of RTSP_URL

MODEL_PATH = r"D:\Rushikesh\project\CNN\LSTM-train\models\resume\model_np_finetuned.pth"

# IMPORTANT: model only predicts these two classes
CLASSES = ["normal", "picked"]

SEQ_LEN = 16
IMG_SIZE = 224
INFER_EVERY = 8  # run model every N frames (lower = more frequent inference)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# smoothing window (in number of inference outputs)
SMOOTH_WINDOW = 8

# class thresholds (for the two model classes)
CLASS_THRESHOLDS = {"normal": 0.50, "picked": 0.60}

# FSM timing (seconds)
MIN_PERSIST = {("normal", "picked"): 0.5, ("picked", "normal"): 0.6}

# SIMPLE PLACED LOGIC: When picked -> normal (committed), show placed for PLACED_DURATION seconds
PLACED_DURATION = 2.0  # seconds to show 'placed'

# PICKED LOCK (HYSTERESIS)
PICKED_LOCK_TIME = 1.5  # seconds (prevents quick flicker out of 'picked')

# SPEED TUNING
USE_FP16 = True if (DEVICE == "cuda") else False   # use half precision on GPU for faster inference
SET_OPS_THREADS = 1    # OpenCV threads (lower overhead)
CAP_PROP_BUFFERSIZE = 1  # reduce internal capture buffer

# display colors
COLORS = {"normal": (200, 200, 200), "picked": (0, 255, 0), "placed": (0, 165, 255)}

# UI
WINDOW_NAME = "Kalmar FSM Action Detection (RTSP fast)"
BUTTON_W = 140
BUTTON_H = 50
BUTTON_MARGIN = 20
SKIP_SECONDS = 2.0

# ============================================================
# MODEL (same architecture as your training)
# ============================================================
class CNN_LSTM_Industry(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.cnn = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.cnn_fc = nn.Sequential(
            nn.Linear(1280, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.lstm = nn.LSTM(input_size=768, hidden_size=256, num_layers=1, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        f = self.cnn(x)
        f = self.pool(f).view(f.size(0), -1)
        f = self.cnn_fc(f)
        f = f.view(B, T, -1)
        out, _ = self.lstm(f)
        out = out.mean(dim=1)
        return self.classifier(out)

# ============================================================
# FAST PREPROCESS (cv2 -> torch tensor) — avoids PIL overhead
# ============================================================
# Normalization used during training:
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_frame_fast(frame_bgr):
    # frame_bgr: HxWx3 BGR (OpenCV)
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    # HWC -> CHW
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.from_numpy(img)  # float32
    return tensor

# ============================================================
# THREAD-BASED FRAME GRABBER (keeps latest frame only)
# ============================================================
class FrameGrabber(threading.Thread):
    def __init__(self, src, use_ffmpeg=True):
        super().__init__(daemon=True)
        self.src = src
        self.keep_running = True
        self.latest_frame = None
        # Open capture in the thread to avoid main thread blocking on open
        self.cap = None
        self.use_ffmpeg = use_ffmpeg

    def open(self):
        # tune OpenCV threads and backend
        cv2.setNumThreads(SET_OPS_THREADS)
        if self.use_ffmpeg:
            self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
        else:
            self.cap = cv2.VideoCapture(self.src)
        if self.cap is None or not self.cap.isOpened():
            return False
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, CAP_PROP_BUFFERSIZE)
        except Exception:
            pass
        return True

    def run(self):
        if not self.open():
            print("❌ FrameGrabber: cannot open source:", self.src)
            self.keep_running = False
            return
        while self.keep_running:
            ret, frame = self.cap.read()
            if not ret:
                # small sleep to avoid tight loop when stream lost
                time.sleep(0.01)
                continue
            # store the latest frame (drop older)
            self.latest_frame = frame
        # cleanup
        if self.cap is not None:
            self.cap.release()

    def stop(self):
        self.keep_running = False

# ============================================================
# TEMPORAL SMOOTHER (same logic as before)
# ============================================================
from collections import deque as _dq

class TemporalSmoother:
    def __init__(self, classes, window=8):
        self.window = window
        self.buffer = _dq(maxlen=window)
        self.classes = classes
        self.last_change_time = None
        self.last_smooth = None

    def update(self, pred, probs):
        self.buffer.append((pred, probs.copy()))
        if self.last_smooth is None:
            self.last_smooth = pred
            self.last_change_time = time.time()

        if len(self.buffer) < self.window:
            conf = float(probs[self.classes.index(pred)])
            if pred != self.last_smooth:
                self.last_smooth = pred
                self.last_change_time = time.time()
            return pred, conf

        score = {c: 0.0 for c in self.classes}
        count = {c: 0 for c in self.classes}
        for p, pr in self.buffer:
            count[p] += 1
            for i, c in enumerate(self.classes):
                score[c] += float(pr[i])

        best_by_score = max(score.items(), key=lambda x: x[1])[0]
        best_by_count = max(count.items(), key=lambda x: x[1])[0]

        final = best_by_score
        final_conf = score[final] / float(len(self.buffer))

        if final_conf < 0.3:
            final = best_by_count
            final_conf = score[final] / float(len(self.buffer))

        if final != self.last_smooth:
            self.last_smooth = final
            self.last_change_time = time.time()

        return final, float(final_conf)

# ============================================================
# FSM with picked-lock + placed transient (keeps same behavior)
# ============================================================
class KalmarFSM:
    def __init__(self):
        self.state = "normal"
        self.state_enter_time = time.time()
        self.placed_active = False
        self.placed_start_time = None
        self.last_fsm_state = "normal"

    def can_transition(self, from_s, to_s):
        if self.placed_active:
            return False
        if from_s == "normal" and to_s == "picked":
            return True
        if from_s == "picked" and to_s == "normal":
            # enforce lock/hysteresis
            if (time.time() - self.state_enter_time) < PICKED_LOCK_TIME:
                return False
            return True
        return False

    def force_state(self, s):
        self.last_fsm_state = self.state
        self.state = s
        self.state_enter_time = time.time()

    def trigger_placed(self):
        self.placed_active = True
        self.placed_start_time = time.time()
        print(f"[FSM] Triggered 'placed' for {PLACED_DURATION} seconds")

    def update_placed(self):
        if self.placed_active and (time.time() - self.placed_start_time >= PLACED_DURATION):
            self.placed_active = False
            self.force_state("normal")
            print("[FSM] Placed phase ended, returning to 'normal'")
            return True
        return False

# ============================================================
# LOAD MODEL
# ============================================================
print("🔹 Loading model...")
model = CNN_LSTM_Industry(len(CLASSES)).to(DEVICE)

ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
# warn if classes metadata exists
if "classes" in ckpt and ckpt["classes"] != CLASSES:
    print("⚠️ Warning: checkpoint 'classes' metadata differs from current CLASSES.")
    print("  checkpoint classes:", ckpt["classes"])
    print("  using current CLASSES:", CLASSES)
model.load_state_dict(ckpt["model"], strict=True)
model.eval()

if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True
    if USE_FP16:
        try:
            model.half()
            print("⚡ Using FP16 (half) inference on GPU")
        except Exception as e:
            print("⚠️ Failed to convert model to half precision:", e)

print("✅ Model loaded")

# ============================================================
# Setup capture (RTSP or file) — use threaded grabber
# ============================================================
source = RTSP_URL if (VIDEO_PATH is None) else VIDEO_PATH
use_ffmpeg = True if (VIDEO_PATH is None) else False
grabber = FrameGrabber(source, use_ffmpeg=use_ffmpeg)
grabber.start()

# wait for first frame
t0 = time.time()
while grabber.latest_frame is None and (time.time() - t0 < 10):
    time.sleep(0.05)
if grabber.latest_frame is None:
    grabber.stop()
    raise RuntimeError("❌ Could not get any frame from source. Check RTSP_URL or VIDEO_PATH.")

# preparation
buffer = deque(maxlen=SEQ_LEN)
smoother = TemporalSmoother(CLASSES, window=SMOOTH_WINDOW)
fsm = KalmarFSM()
frame_idx = 0
last_conf = 0.0

# UI
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
_mouse_click = {"x": -1, "y": -1, "clicked": False}
def mouse_callback(event, x, y, flags, param):
    global _mouse_click
    if event == cv2.EVENT_LBUTTONDOWN:
        _mouse_click = {"x": x, "y": y, "clicked": True}
cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

print("▶ Running inference (fast RTSP mode). Core logic: picked -> normal => placed for", PLACED_DURATION, "sec")
if VIDEO_PATH is None:
    print("Source:", RTSP_URL)
else:
    print("Source file:", VIDEO_PATH)

# ===================== MAIN LOOP =====================
try:
    while True:
        frame = grabber.latest_frame
        if frame is None:
            # small wait if stream stalled
            time.sleep(0.005)
            continue

        frame_idx += 1

        # append preprocessed frame to buffer (fast numpy->tensor)
        t = preprocess_frame_fast(frame)  # float32 torch tensor CHW
        if USE_FP16 and DEVICE == "cuda":
            t = t.half()
        buffer.append(t)

        # inference condition
        if (frame_idx % INFER_EVERY == 0) and (len(buffer) == SEQ_LEN):
            # build batch (1, T, C, H, W)
            x = torch.stack(list(buffer)).unsqueeze(0).to(DEVICE)
            if USE_FP16 and DEVICE == "cuda":
                x = x.half()

            with torch.inference_mode():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

            best_idx = int(np.argmax(probs))
            best_class = CLASSES[best_idx]
            best_conf = float(probs[best_idx])
            predicted = best_class if best_conf >= CLASS_THRESHOLDS[best_class] else "normal"

            smooth_pred, smooth_conf = smoother.update(predicted, probs)

            # update placed expiry first
            fsm.update_placed()

            # transition gating using smoothed prediction
            if smoother.last_change_time is None:
                smoother.last_change_time = time.time()
                smoother.last_smooth = smooth_pred
            time_persisted = time.time() - smoother.last_change_time

            if fsm.can_transition(fsm.state, smooth_pred):
                required = MIN_PERSIST.get((fsm.state, smooth_pred), 0.5)
                if time_persisted >= required:
                    prev_state = fsm.state
                    fsm.force_state(smooth_pred)
                    print(f"[FSM] committed transition {prev_state} -> {fsm.state}")

                    # trigger placed only when committed picked -> normal
                    if prev_state == "picked" and fsm.state == "normal":
                        fsm.trigger_placed()

            # display state selection
            if fsm.placed_active:
                display_state = "placed"
                last_conf = smooth_conf
            else:
                display_state = fsm.state
                last_conf = smooth_conf

            print(f"Frame {frame_idx:05d} | Raw={predicted:7s} | Smooth={smooth_pred:7s} ({smooth_conf:.2f}) | FSM={fsm.state:7s} | Display={display_state:7s}")

        # ---------------- UI overlay & show ----------------
        display = frame.copy()
        h, w = display.shape[:2]

        # draw simple state text and confidence
        if fsm.placed_active:
            placed_elapsed = time.time() - fsm.placed_start_time
            placed_remaining = max(0, PLACED_DURATION - placed_elapsed)
            state_text = f"STATE: PLACED ({placed_remaining:.1f}s)"
            color = COLORS["placed"]
        else:
            state_text = f"STATE: {fsm.state.upper()}"
            color = COLORS.get(fsm.state, (220,220,220))

        cv2.putText(display, state_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        cv2.putText(display, f"CONF: {last_conf:.2f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        info_text = f"Frame: {frame_idx}  FPS(src): {round(1/(1e-6 + (time.time()-grabber.cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0)) ,1) if grabber.cap is not None else '?'}"
        # (pos-based FPS estimation can be noisy; we display minimal info)
        cv2.putText(display, f"Source: {'RTSP' if VIDEO_PATH is None else 'FILE'}", (30, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(1) & 0xFF
        if key != 255:
            if key == ord('q'):
                break
            if key == ord('f'):
                # forward skip
                if grabber.cap is not None:
                    skip_frames = int(SKIP_SECONDS * (grabber.cap.get(cv2.CAP_PROP_FPS) or 25.0))
                    target = max(0, int(grabber.cap.get(cv2.CAP_PROP_POS_FRAMES)) + skip_frames)
                    grabber.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            if key == ord('b'):
                if grabber.cap is not None:
                    skip_frames = int(SKIP_SECONDS * (grabber.cap.get(cv2.CAP_PROP_FPS) or 25.0))
                    target = max(0, int(grabber.cap.get(cv2.CAP_PROP_POS_FRAMES)) - skip_frames)
                    grabber.cap.set(cv2.CAP_PROP_POS_FRAMES, target)

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    grabber.stop()
    cv2.destroyAllWindows()
    print("✅ Finished")
