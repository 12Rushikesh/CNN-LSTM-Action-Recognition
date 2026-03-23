# ============================================================
# Kalmar Action Detection — Live Inference
# CNN + LSTM (pretrained) + Temporal Smoothing + FSM + Transient Placed
# Rewritten to integrate smoothing (majority + confidence) and a robust
# FSM suitable for industrial-style inference.
# Save this file and run it in the same environment where your model and
# video files are available.
# ============================================================

import cv2
import torch
import numpy as np
import torchvision.transforms as T
import torchvision.models as models
import torch.nn as nn
from collections import deque, Counter
import time

# ---------------- Windows / LSTM safety ----------------
torch.backends.cudnn.enabled = False

# ===================== CONFIG =====================
MODEL_PATH = r"D:\Rushikesh\project\CNN\LSTM-train\models\model2\4.pth"
VIDEO_PATH = r"C:\Users\Rapportsoft\Documents\kalmardata\10.mp4"

CLASSES = ["normal", "picked", "placed"]
SEQ_LEN = 16
IMG_SIZE = 224
INFER_EVERY = 8  # run model every N frames
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# smoothing window (in number of inference outputs, not frames)
SMOOTH_WINDOW = 8

# class thresholds (same idea as your original)
CLASS_THRESHOLDS = {"normal": 0.45, "picked": 0.35, "placed": 0.65}

# FSM timing (seconds)
MIN_PERSIST = {  # minimum time that a smoothed prediction must persist before we allow transition
    ("normal", "picked"): 0.5,
    ("picked", "placed"): 0.5,
    ("placed", "normal"): 1.0
}

WAIT_BEFORE_TEMP = 5.0
TEMP_PLACED_DURATION = 2.0

# display colors
COLORS = {"normal": (200, 200, 200), "picked": (0, 255, 0), "placed": (0, 165, 255)}

# UI
WINDOW_NAME = "Kalmar FSM Action Detection"
BUTTON_W = 140
BUTTON_H = 50
BUTTON_MARGIN = 20
SKIP_SECONDS = 2.0

# ===================== MODEL (same as training) =====================
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

# ===================== SMOOTHERS =====================
class TemporalSmoother:
    """Maintains a short history of model outputs (pred, probs) and returns
    a smoothed decision. Uses both confidence-sum and majority voting.
    """
    def __init__(self, classes, window=8):
        self.window = window
        self.buffer = deque(maxlen=window)
        self.classes = classes

    def update(self, pred, probs):
        # pred: string label, probs: np.array same order as CLASSES
        self.buffer.append((pred, probs.copy()))

        if len(self.buffer) < self.window:
            # not enough history -> return immediate pred + its confidence
            return pred, float(probs[self.classes.index(pred)])

        # confidence-sum voting
        score = {c: 0.0 for c in self.classes}
        count = {c: 0 for c in self.classes}
        for p, pr in self.buffer:
            count[p] += 1
            for i, c in enumerate(self.classes):
                score[c] += float(pr[i])

        # pick by highest confidence-sum, break ties with majority count
        best_by_score = max(score.items(), key=lambda x: x[1])[0]
        best_by_count = max(count.items(), key=lambda x: x[1])[0]

        # final selection: prefer best_by_score but ensure it isn't an extremely low score
        final = best_by_score
        final_conf = score[final] / float(len(self.buffer))

        # if the score winner is weak but majority prefers another, prefer majority
        if final_conf < 0.3:
            final = best_by_count
            # estimate confidence as avg of that class
            final_conf = score[final] / float(len(self.buffer))

        return final, float(final_conf)

# ===================== FSM CLASS =====================
class KalmarFSM:
    def __init__(self, classes):
        self.state = "normal"
        self.state_enter_time = time.time()
        self.allowed = {"normal": ["picked"], "picked": ["placed"], "placed": ["normal"]}

        # transient placed logic
        self.pick_normal_timer_start = None
        self.placed_once_after_picked = False
        self.temp_placed_active = False
        self.temp_placed_start_time = None

    def can_transition(self, from_s, to_s):
        return to_s in self.allowed.get(from_s, [])

    def time_in_state(self):
        return time.time() - self.state_enter_time

    def try_transition(self, candidate, candidate_conf):
        """Attempt to transition to candidate state. We use MIN_PERSIST to
        enforce a minimal persistence of the candidate prediction.
        candidate: str label
        candidate_conf: float
        Returns True if state changed, False otherwise.
        """
        if candidate == self.state:
            # reset timers if the same
            self.state_enter_time = self.state_enter_time  # no-op but clear intent
            return False

        if not self.can_transition(self.state, candidate):
            return False

        # required minimal duration for the proposed transition
        key = (self.state, candidate)
        required = MIN_PERSIST.get(key, 0.5)

        # we don't have an internal persistence counter for candidate here;
        # the caller should feed smoothed decisions repeatedly. We'll require
        # that candidate_conf is decent and that time_in_state >= 0 (we will
        # rely on caller to call this only after persistence window)
        if self.time_in_state() >= 0:  # placeholder — actual gating done by caller
            # perform immediate transition; caller should ensure candidate persisted
            prev = self.state
            self.state = candidate
            self.state_enter_time = time.time()
            # when leaving picked reset transient latch
            if prev != "picked" and self.state == "picked":
                self.placed_once_after_picked = False
            return True

        return False

    def force_state(self, s):
        self.state = s
        self.state_enter_time = time.time()

# ===================== LOAD MODEL =====================
print("🔹 Loading model...")
model = CNN_LSTM_Industry(len(CLASSES)).to(DEVICE)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model"], strict=True)
model.eval()
print("✅ Model loaded")

# ===================== TRANSFORM =====================
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ===================== VIDEO SETUP =====================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Cannot open video")

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

buffer = deque(maxlen=SEQ_LEN)
frame_idx = 0

smoother = TemporalSmoother(CLASSES, window=SMOOTH_WINDOW)
fsm = KalmarFSM(CLASSES)

last_conf = 0.0

# UI callbacks
_mouse_click = {"x": -1, "y": -1, "clicked": False}

def mouse_callback(event, x, y, flags, param):
    global _mouse_click
    if event == cv2.EVENT_LBUTTONDOWN:
        _mouse_click = {"x": x, "y": y, "clicked": True}

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

seeking = False
inference_done_since_seek = False

print("▶ Running inference with smoothing + FSM")

# helper: jump frames
def jump_frames(cap_obj, target_frame_idx):
    global buffer, frame_idx, seeking, inference_done_since_seek
    if total_frames > 0:
        target_frame_idx = int(max(0, min(total_frames - 1, int(target_frame_idx))))
    else:
        target_frame_idx = int(max(0, int(target_frame_idx)))
    cap_obj.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
    buffer.clear()
    seeking = True
    inference_done_since_seek = False
    frame_idx = target_frame_idx
    return frame_idx

# main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # update frame_idx
    pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    if pos is not None:
        frame_idx = max(0, int(pos) - 1)

    display = frame.copy()
    h, w = display.shape[:2]

    # draw UI buttons
    left_btn_tl = (BUTTON_MARGIN, h - BUTTON_H - BUTTON_MARGIN)
    left_btn_br = (BUTTON_MARGIN + BUTTON_W, h - BUTTON_MARGIN)
    right_btn_tl = (w - BUTTON_MARGIN - BUTTON_W, h - BUTTON_H - BUTTON_MARGIN)
    right_btn_br = (w - BUTTON_MARGIN, h - BUTTON_MARGIN)

    cv2.rectangle(display, left_btn_tl, left_btn_br, (50, 50, 50), -1)
    cv2.rectangle(display, left_btn_tl, left_btn_br, (200, 200, 200), 2)
    cv2.putText(display, "BACK", (left_btn_tl[0] + 15, left_btn_tl[1] + BUTTON_H // 2 + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.rectangle(display, right_btn_tl, right_btn_br, (50, 50, 50), -1)
    cv2.rectangle(display, right_btn_tl, right_btn_br, (200, 200, 200), 2)
    cv2.putText(display, "FORWARD", (right_btn_tl[0] + 8, right_btn_tl[1] + BUTTON_H // 2 + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # preprocess frame
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    buffer.append(transform(rgb))

    # show seeking progress UI
    show_fsm_state = True
    if seeking:
        show_fsm_state = False
        prog_text = f"SEEKING... buffer: {len(buffer)}/{SEQ_LEN}"
        cv2.putText(display, prog_text, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 200), 2)
        bar_w = int((w - 60) * (len(buffer) / SEQ_LEN))
        cv2.rectangle(display, (30, 120), (30 + (w - 60), 140), (80, 80, 80), -1)
        cv2.rectangle(display, (30, 120), (30 + bar_w, 140), (0, 200, 200), -1)

    inference_ran = False
    if frame_idx % INFER_EVERY == 0 and len(buffer) == SEQ_LEN:
        x = torch.stack(list(buffer)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        inference_ran = True

        # threshold logic (same as yours but we keep full probs for smoothing)
        best_idx = int(np.argmax(probs))
        best_class = CLASSES[best_idx]
        best_conf = float(probs[best_idx])

        predicted = "normal"
        if best_conf >= CLASS_THRESHOLDS[best_class]:
            predicted = best_class
        else:
            placed_idx = CLASSES.index("placed")
            if probs[placed_idx] >= CLASS_THRESHOLDS["placed"]:
                predicted = "placed"

        # SMOOTHING: update smoother with raw prediction and full probs
        smooth_pred, smooth_conf = smoother.update(predicted, probs)

        # ---------- Transient picked->normal logic ----------
        # Start/stop timer when we're in picked state and smoothed says normal
        if fsm.state == "picked":
            if smooth_pred == "normal" and not fsm.placed_once_after_picked:
                if fsm.pick_normal_timer_start is None:
                    fsm.pick_normal_timer_start = time.time()
                else:
                    elapsed = time.time() - fsm.pick_normal_timer_start
                    if elapsed >= WAIT_BEFORE_TEMP and not fsm.temp_placed_active:
                        fsm.temp_placed_active = True
                        fsm.temp_placed_start_time = time.time()
                        fsm.placed_once_after_picked = True
                        print(f"[transient] picked->normal persisted {WAIT_BEFORE_TEMP}s — showing 'placed' ONCE")
            else:
                fsm.pick_normal_timer_start = None
        else:
            fsm.pick_normal_timer_start = None

        # ---------- Handle transient expiry ----------
        if fsm.temp_placed_active:
            if time.time() - fsm.temp_placed_start_time >= TEMP_PLACED_DURATION:
                fsm.temp_placed_active = False
                # force FSM to NORMAL to complete cycle
                if fsm.state != "normal":
                    fsm.force_state("normal")
                    fsm.placed_once_after_picked = False
                    fsm.pick_normal_timer_start = None
                    print("[transient] placed ended — FSM forced to NORMAL")

        # ---------- FSM transition gating ----------
        # We require the smoothed prediction to persist for a minimal duration
        # before allowing the transition. We'll track the last time smooth_pred changed.
        if not hasattr(smoother, 'last_change_time'):
            smoother.last_change_time = time.time()
            smoother.last_smooth = smooth_pred

        if smooth_pred != smoother.last_smooth:
            smoother.last_change_time = time.time()
            smoother.last_smooth = smooth_pred

        time_persisted = time.time() - smoother.last_change_time

        # check allowed transition + persistence
        changed = False
        if fsm.can_transition(fsm.state, smooth_pred):
            required = MIN_PERSIST.get((fsm.state, smooth_pred), 0.5)
            if time_persisted >= required:
                # commit transition
                prev = fsm.state
                fsm.force_state(smooth_pred)
                changed = True
                # reset transient latch when leaving picked
                if prev == "picked" and fsm.state != "picked":
                    fsm.placed_once_after_picked = False

        # prepare last_conf for display (respect transient override)
        if fsm.temp_placed_active:
            last_conf = float(probs[CLASSES.index("placed")])
            visible_state = "placed"
        else:
            last_conf = smooth_conf
            visible_state = fsm.state

        print(f"Frame {frame_idx:05d} | Raw={predicted:7s} | Smooth={smooth_pred:7s} ({smooth_conf:.2f}) | FSM={fsm.state:7s} | Conf={last_conf:.2f}")

        # post-seek handling
        if seeking and len(buffer) == SEQ_LEN:
            inference_done_since_seek = True
            seeking = False
            print("[seek] Buffer refilled and first inference done — resuming FSM display")

    # DISPLAY
    if not seeking:
        if fsm.temp_placed_active:
            disp_state = "placed"
        else:
            disp_state = fsm.state
        cv2.putText(display, f"STATE: {disp_state.upper()}  CONF: {last_conf:.2f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLORS[disp_state], 3)

    # show pending timer
    if fsm.pick_normal_timer_start is not None and not fsm.temp_placed_active:
        elapsed = time.time() - fsm.pick_normal_timer_start
        remain = max(0.0, WAIT_BEFORE_TEMP - elapsed)
        timer_text = f"pending placed in: {remain:.2f}s"
        cv2.putText(display, timer_text, (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 50), 2)

    info_text = f"Frame: {frame_idx}/{total_frames if total_frames>0 else '?'}  FPS: {fps:.1f}"
    cv2.putText(display, info_text, (30, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

    cv2.imshow(WINDOW_NAME, display)

    # mouse handling
    if _mouse_click["clicked"]:
        mx, my = _mouse_click["x"], _mouse_click["y"]
        _mouse_click["clicked"] = False
        if left_btn_tl[0] <= mx <= left_btn_br[0] and left_btn_tl[1] <= my <= left_btn_br[1]:
            skip_frames = int(SKIP_SECONDS * fps)
            target = frame_idx - skip_frames
            new_idx = jump_frames(cap, target)
            print(f"[button] BACK clicked -> jumped to frame {new_idx}")
            continue
        if right_btn_tl[0] <= mx <= right_btn_br[0] and right_btn_tl[1] <= my <= right_btn_br[1]:
            skip_frames = int(SKIP_SECONDS * fps)
            target = frame_idx + skip_frames
            new_idx = jump_frames(cap, target)
            print(f"[button] FORWARD clicked -> jumped to frame {new_idx}")
            continue

    key = cv2.waitKey(1) & 0xFF
    if key != 255:
        if key == ord('q'):
            break
        if key == ord('f'):
            skip_frames = int(SKIP_SECONDS * fps)
            target = frame_idx + skip_frames
            new_idx = jump_frames(cap, target)
            print(f"[key] 'f' pressed -> jumped to frame {new_idx}")
            continue
        if key == ord('b'):
            skip_frames = int(SKIP_SECONDS * fps)
            target = frame_idx - skip_frames
            new_idx = jump_frames(cap, target)
            print(f"[key] 'b' pressed -> jumped to frame {new_idx}")
            continue
        if key == ord('n'):
            new_idx = jump_frames(cap, frame_idx + 1)
            print(f"[key] 'n' pressed -> stepped to frame {new_idx}")
            continue
        if key == ord('p'):
            new_idx = jump_frames(cap, frame_idx - 1)
            print(f"[key] 'p' pressed -> stepped to frame {new_idx}")
            continue

# cleanup
cap.release()
cv2.destroyAllWindows()
print("✅ Finished")
