# ============================================================
# Kalmar Action Detection — Live Inference (MODEL predicts only normal/picked)
# CNN + LSTM (pretrained) + Temporal Smoothing + FSM + Simple Placed Logic
# CORE LOGIC: "picked → normal ⇒ placed for 2 seconds"
# Added: PICKED_LOCK_TIME (hysteresis / state-lock) to avoid flicker.
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
MODEL_PATH = r"D:\Rushikesh\project\CNN\LSTM-train\models\2second\Finetuned12\f1.pth"
VIDEO_PATH = r"F:\AI projects\kalmardata\12.mp4"

# IMPORTANT: model only predicts these two classes
CLASSES = ["normal", "picked"]

SEQ_LEN = 16
IMG_SIZE = 224
INFER_EVERY = 8  # run model every N frames
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# smoothing window (in number of inference outputs, not frames)
SMOOTH_WINDOW = 8

# class thresholds (for the two model classes)
CLASS_THRESHOLDS = {"normal": 0.50, "picked": 0.50}

# FSM timing (seconds)
MIN_PERSIST = {  # minimal persistence for committing transitions (seconds)
    ("normal", "picked"): 0.5,
    ("picked", "normal"): 0.6,  # For smooth transition
}

# SIMPLE PLACED LOGIC: When picked→normal (committed), show placed for PLACED_DURATION seconds
PLACED_DURATION = 2.0  # seconds to show 'placed'

# NEW: PICKED LOCK (HYSTERESIS) — prevents premature exit from picked
# Tune this between 1.0 and 2.5 seconds depending on crane dynamics
PICKED_LOCK_TIME = 1.5  # seconds

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
        self.last_change_time = None
        self.last_smooth = None

    def update(self, pred, probs):
        # pred: string label, probs: np.array same order as classes
        self.buffer.append((pred, probs.copy()))

        # initialize last_smooth / last_change_time
        if self.last_smooth is None:
            self.last_smooth = pred
            self.last_change_time = time.time()

        if len(self.buffer) < self.window:
            # not enough history -> return immediate pred + its confidence
            conf = float(probs[self.classes.index(pred)])
            # track smoothing change
            if pred != self.last_smooth:
                self.last_smooth = pred
                self.last_change_time = time.time()
            return pred, conf

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

        final = best_by_score
        final_conf = score[final] / float(len(self.buffer))

        # if the score winner is weak but majority prefers another, prefer majority
        if final_conf < 0.3:
            final = best_by_count
            final_conf = score[final] / float(len(self.buffer))

        # update last_smooth timing
        if final != self.last_smooth:
            self.last_smooth = final
            self.last_change_time = time.time()

        return final, float(final_conf)

# ===================== FSM CLASS =====================
class KalmarFSM:
    def __init__(self, classes):
        self.state = "normal"
        self.state_enter_time = time.time()
        
        # Simple placed logic variables
        self.placed_active = False
        self.placed_start_time = None
        
        # Track last FSM state (before last force)
        self.last_fsm_state = "normal"

    def can_transition(self, from_s, to_s):
        # Block all transitions during placed phase
        if self.placed_active:
            return False

        # Normal transitions (model-predicted states)
        if from_s == "normal" and to_s == "picked":
            return True

        # For picked -> normal, enforce the PICKED_LOCK_TIME hysteresis:
        if from_s == "picked" and to_s == "normal":
            # if still within lock time, refuse
            if (time.time() - self.state_enter_time) < PICKED_LOCK_TIME:
                # still locked - do not allow picked -> normal transition yet
                return False
            return True

        return False

    def time_in_state(self):
        return time.time() - self.state_enter_time

    def force_state(self, s):
        # Update last state before changing
        self.last_fsm_state = self.state
        self.state = s
        self.state_enter_time = time.time()

    def trigger_placed(self):
        """Start the placed phase for PLACED_DURATION seconds"""
        self.placed_active = True
        self.placed_start_time = time.time()
        print(f"[FSM] Triggered 'placed' for {PLACED_DURATION} seconds")

    def update_placed(self):
        """Check if placed phase should end"""
        if self.placed_active and (time.time() - self.placed_start_time >= PLACED_DURATION):
            self.placed_active = False
            # Force back to normal after placed phase
            self.force_state("normal")
            print("[FSM] Placed phase ended, returning to 'normal'")
            return True
        return False

# ===================== LOAD MODEL =====================
print("🔹 Loading model...")
model = CNN_LSTM_Industry(len(CLASSES)).to(DEVICE)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
# if checkpoint has 'classes' metadata, warn if differs
if "classes" in ckpt and ckpt["classes"] != CLASSES:
    print("⚠️ Warning: checkpoint 'classes' metadata differs from current CLASSES.")
    print("  checkpoint classes:", ckpt["classes"])
    print("  using current CLASSES:", CLASSES)
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
print(f"📌 CORE LOGIC: picked → normal ⇒ placed for {PLACED_DURATION} seconds")
print(f"📌 PICKED_LOCK_TIME: {PICKED_LOCK_TIME:.2f}s (hysteresis)")

# helper: jump frames
def jump_frames(cap_obj, target_frame_idx):
    global buffer, frame_idx, seeking, inference_done_since_seek, smoother
    if total_frames > 0:
        target_frame_idx = int(max(0, min(total_frames - 1, int(target_frame_idx))))
    else:
        target_frame_idx = int(max(0, int(target_frame_idx)))
    cap_obj.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
    buffer.clear()
    seeking = True
    inference_done_since_seek = False
    frame_idx = target_frame_idx
    # reset smoother state so history rebuilds cleanly
    smoother.buffer.clear()
    smoother.last_smooth = None
    smoother.last_change_time = None
    # Reset FSM to normal when seeking
    fsm.state = "normal"
    fsm.placed_active = False
    fsm.last_fsm_state = "normal"
    fsm.state_enter_time = time.time()
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

        # RAW prediction (model only predicts normal/picked)
        best_idx = int(np.argmax(probs))
        best_class = CLASSES[best_idx]
        best_conf = float(probs[best_idx])

        predicted = best_class if best_conf >= CLASS_THRESHOLDS[best_class] else "normal"

        # SMOOTHING: update smoother with raw prediction and full probs
        smooth_pred, smooth_conf = smoother.update(predicted, probs)

        # ---------- CORE: first update placed expiry if active ----------
        fsm.update_placed()

        # ---------- FSM transition gating using smoothed prediction ----------
        # initialize smoother timing if needed
        if smoother.last_change_time is None:
            smoother.last_change_time = time.time()
            smoother.last_smooth = smooth_pred

        # time persisted for current smooth_pred
        time_persisted = time.time() - smoother.last_change_time

        committed_transition = False
        # allow transitions (normal <-> picked) when smoothed prediction persisted enough
        if fsm.can_transition(fsm.state, smooth_pred):
            required = MIN_PERSIST.get((fsm.state, smooth_pred), 0.5)
            if time_persisted >= required:
                prev_state = fsm.state
                fsm.force_state(smooth_pred)
                committed_transition = True
                print(f"[FSM] committed transition {prev_state} -> {fsm.state}")

                # --- NEW: trigger placed ONLY when transition committed from picked -> normal ---
                if prev_state == "picked" and fsm.state == "normal":
                    # This is where the picked->normal is confirmed: start placed phase
                    fsm.trigger_placed()

        # Prepare display state and confidence
        if fsm.placed_active:
            display_state = "placed"
            # Show remaining placed time
            placed_elapsed = time.time() - fsm.placed_start_time
            placed_remaining = max(0, PLACED_DURATION - placed_elapsed)
            # Use the smoothed confidence from before placed phase
            last_conf = smooth_conf
        else:
            display_state = fsm.state
            last_conf = smooth_conf

        print(f"Frame {frame_idx:05d} | Raw={predicted:7s} | Smooth={smooth_pred:7s} ({smooth_conf:.2f}) | FSM={fsm.state:7s} | Display={display_state:7s}")

        # post-seek handling
        if seeking and len(buffer) == SEQ_LEN:
            inference_done_since_seek = True
            seeking = False
            print("[seek] Buffer refilled and first inference done — resuming FSM display")

    # DISPLAY
    if not seeking:
        # Determine what to display
        if fsm.placed_active:
            disp_state = "placed"
            placed_elapsed = time.time() - fsm.placed_start_time
            placed_remaining = max(0, PLACED_DURATION - placed_elapsed)
            # Add timer to state display
            state_text = f"STATE: {disp_state.upper()} ({placed_remaining:.1f}s)"
        else:
            disp_state = fsm.state
            state_text = f"STATE: {disp_state.upper()}"
        
        cv2.putText(display, state_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLORS[disp_state], 3)
        cv2.putText(display, f"CONF: {last_conf:.2f}", (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLORS[disp_state], 2)

    # Show FSM logic status in corner
    logic_text = f"Logic: {fsm.last_fsm_state}→{smoother.last_smooth if smoother.last_smooth else 'N/A'}"
    cv2.putText(display, logic_text, (w - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

    info_text = f"Frame: {frame_idx}/{total_frames if total_frames>0 else '?'}  FPS: {fps:.1f}"
    cv2.putText(display, info_text, (30, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

    cv2.imshow(WINDOW_NAME, display)

    # mouse handling for BACK/FORWARD
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

    # keyboard handling
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
