# ============================================================
# KALMAR ACTION DETECTION SYSTEM
# CNN + LSTM + CLASS THRESHOLDS + FSM (STATE MACHINE)
# + Forward / Backward buttons (seek)
# + SEEKING overlay & buffer-progress
# + special: normal -> picked -> normal => wait 2s, then show 'placed' for 2s (transient)
#   (shows only once per picked->normal cycle)
# + Fix: when transient 'placed' ends, force FSM to NORMAL to complete the cycle
# ============================================================

import cv2
import torch
import numpy as np
import torchvision.transforms as T
import torchvision.models as models
import torch.nn as nn
from collections import deque
import time

# ------------------------------------------------------------
# WINDOWS / CUDA SAFETY
# Prevent LSTM cuDNN crash on Windows
# ------------------------------------------------------------
torch.backends.cudnn.enabled = False

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = r"D:\Rushikesh\project\CNN\LSTM-train\models\model2\4.pth"
VIDEO_PATH = r"C:\Users\Rapportsoft\Downloads\kalmar4.mp4"

CLASSES = ["normal", "picked", "placed"]

SEQ_LEN = 16              # must match training
IMG_SIZE = 224            # must match training
INFER_EVERY = 8           # inference frequency (frames)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------
# CLASS-SPECIFIC CONFIDENCE THRESHOLDS
# ------------------------------------------------------------
CLASS_THRESHOLDS = {
    "normal": 0.55,
    "picked": 0.55,
    "placed": 0.70   # lower threshold for rare class
}

# Display colors
COLORS = {
    "normal": (200, 200, 200),
    "picked": (0, 255, 0),
    "placed": (0, 165, 255)
}

# ============================================================
# FSM (FINITE STATE MACHINE)
# ============================================================

ALLOWED_TRANSITIONS = {
    "normal": ["picked"],
    "picked": ["placed"],
    "placed": ["normal"]
}

current_state = "normal"
state_counter = 0
MIN_STATE_FRAMES = 6   # stability window (unchanged)


def fsm_update(current, predicted):
    """
    Enforce valid Kalmar workflow transitions.
    """
    if predicted == current:
        return current
    if predicted in ALLOWED_TRANSITIONS[current]:
        return predicted
    return current


# ============================================================
# MODEL DEFINITION (EXACT SAME AS TRAINING)
# ============================================================

class CNN_LSTM_Industry(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        self.cnn = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.cnn_fc = nn.Sequential(
            nn.Linear(1280, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            dropout=0.0
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: (B, T, C, H, W)
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
# LOAD TRAINED MODEL
# ============================================================

print("🔹 Loading model...")
model = CNN_LSTM_Industry(len(CLASSES)).to(DEVICE)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model"], strict=True)

model.eval()
print("✅ Model loaded successfully")

# ============================================================
# IMAGE TRANSFORM (MATCH TRAINING)
# ============================================================

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# SPECIAL TRANSIENT-PLACED CONDITION (UPDATED)
# Behavior:
#  - If the FSM is in 'picked' and inference predicts 'normal',
#    start a WAIT_BEFORE_TEMP timer (2.0s).
#  - If 'normal' predictions persist for WAIT_BEFORE_TEMP seconds,
#    activate a transient 'placed' display for TEMP_PLACED_DURATION seconds (2.0s).
#  - Show this transient ONLY ONCE per picked->normal cycle.
#  - AFTER the transient ends we FORCE the FSM to normal to complete the cycle.
# ============================================================
WAIT_BEFORE_TEMP = 5.0         # seconds of continuous 'normal' after 'picked' before showing transient 'placed'
TEMP_PLACED_DURATION = 2.0     # seconds to show transient 'placed'
temp_placed_active = False
temp_placed_start_time = 0.0
picked_normal_start_time = None  # None when not timing picked->normal
display_override = None  # when set to "placed" we show that instead of current_state

# ONE-TIME LATCH: ensures the transient 'placed' shows only once per picked->normal cycle
placed_once_after_picked = False

# ============================================================
# FORWARD / BACKWARD BUTTONS CONFIG
# ============================================================
SKIP_SECONDS = 2.0  # click skip duration in seconds
WINDOW_NAME = "Kalmar FSM Action Detection"

# ============================================================
# SEEK/BUFFER STATE (prevents "always normal" after seek)
# ============================================================
seeking = False
inference_done_since_seek = False

# ============================================================
# VIDEO LOOP (FSM-BASED INFERENCE) + Mouse Controls + SEEK UI
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("❌ Cannot open video")

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
buffer = deque(maxlen=SEQ_LEN)
frame_idx = 0
last_conf = 0.0

# Button geometry
BUTTON_W = 140
BUTTON_H = 50
BUTTON_MARGIN = 20

_mouse_click = {"x": -1, "y": -1, "clicked": False}


def mouse_callback(event, x, y, flags, param):
    global _mouse_click
    if event == cv2.EVENT_LBUTTONDOWN:
        _mouse_click = {"x": x, "y": y, "clicked": True}


cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

print("▶ Running FSM-based inference (click buttons or press 'f'/'b' to seek)")

def jump_frames(cap_obj, target_frame_idx):
    """
    Safely jump to a target frame index and prepare seeking state.
    Clears buffer so temporal sequence rebuilds from new position.
    """
    global buffer, frame_idx, temp_placed_active, display_override, seeking, inference_done_since_seek, picked_normal_start_time, placed_once_after_picked
    if total_frames > 0:
        target_frame_idx = int(max(0, min(total_frames - 1, int(target_frame_idx))))
    else:
        target_frame_idx = int(max(0, int(target_frame_idx)))
    cap_obj.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
    buffer.clear()  # rebuild temporal window from scratch
    temp_placed_active = False
    display_override = None
    picked_normal_start_time = None
    placed_once_after_picked = False
    seeking = True
    inference_done_since_seek = False
    frame_idx = target_frame_idx
    return frame_idx


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Update frame_idx from capture position (safer when seeking)
    pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    if pos is not None:
        frame_idx = max(0, int(pos) - 1)

    display = frame.copy()
    h, w = display.shape[:2]

    # compute buttons' positions each frame
    left_btn_tl = (BUTTON_MARGIN, h - BUTTON_H - BUTTON_MARGIN)
    left_btn_br = (BUTTON_MARGIN + BUTTON_W, h - BUTTON_MARGIN)
    right_btn_tl = (w - BUTTON_MARGIN - BUTTON_W, h - BUTTON_H - BUTTON_MARGIN)
    right_btn_br = (w - BUTTON_MARGIN, h - BUTTON_MARGIN)

    # draw buttons
    cv2.rectangle(display, left_btn_tl, left_btn_br, (50, 50, 50), -1)
    cv2.rectangle(display, left_btn_tl, left_btn_br, (200, 200, 200), 2)
    cv2.putText(display, "BACK", (left_btn_tl[0] + 15, left_btn_tl[1] + BUTTON_H // 2 + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.rectangle(display, right_btn_tl, right_btn_br, (50, 50, 50), -1)
    cv2.rectangle(display, right_btn_tl, right_btn_br, (200, 200, 200), 2)
    cv2.putText(display, "FORWARD", (right_btn_tl[0] + 8, right_btn_tl[1] + BUTTON_H // 2 + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Preprocess frame for buffer
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    buffer.append(transform(rgb))

    # If transient placed active, check expiry and FORCE FSM completion when it ends
    if temp_placed_active:
        if time.time() - temp_placed_start_time >= TEMP_PLACED_DURATION:
            temp_placed_active = False
            display_override = None

            # ===== FIX: Force FSM to NORMAL to complete the cycle =====
            # This makes the transient 'placed' behave like a real placed event
            # and prevents the FSM from remaining stuck in 'picked'.
            if current_state != "normal":
                current_state = "normal"
                state_counter = 0
                placed_once_after_picked = False
                picked_normal_start_time = None
                print("[transient] placed ended — FSM forced to NORMAL")
            # =========================================================

    # -------------------------
    # SEEK UX: while seeking, hide FSM state until buffer fills & a post-seek inference runs
    # -------------------------
    show_fsm_state = True
    if seeking:
        show_fsm_state = False
        prog_text = f"SEEKING... buffer: {len(buffer)}/{SEQ_LEN}"
        cv2.putText(display, prog_text, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 200), 2)
        bar_w = int((w - 60) * (len(buffer) / SEQ_LEN))
        cv2.rectangle(display, (30, 120), (30 + (w - 60), 140), (80, 80, 80), -1)
        cv2.rectangle(display, (30, 120), (30 + bar_w, 140), (0, 200, 200), -1)
        # if buffer full, wait for inference frame to run below to clear seeking flag

    # --------------------------------------------------------
    # INFERENCE STEP (original logic preserved, plus the waiting->transient behavior)
    # --------------------------------------------------------
    inference_ran = False
    if frame_idx % INFER_EVERY == 0 and len(buffer) == SEQ_LEN:
        x = torch.stack(list(buffer)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        inference_ran = True

        # -------- THRESHOLD DECISION (unchanged) --------
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

        # ------------------------
        # NEW: handle picked -> normal waiting logic (ONE-TIME latch)
        # ------------------------
        if current_state == "picked":
            if predicted == "normal" and not placed_once_after_picked:
                if picked_normal_start_time is None:
                    # start timing the continuous normal observation
                    picked_normal_start_time = time.time()
                else:
                    # check if normal persisted long enough
                    elapsed = time.time() - picked_normal_start_time
                    if elapsed >= WAIT_BEFORE_TEMP and not temp_placed_active:
                        # Activate transient placed display for TEMP_PLACED_DURATION
                        temp_placed_active = True
                        temp_placed_start_time = time.time()
                        display_override = "placed"
                        placed_once_after_picked = True   # latch so it won't repeat
                        picked_normal_start_time = None  # reset timer after activation
                        print(f"[transient] picked->normal persisted {WAIT_BEFORE_TEMP}s — showing 'placed' ONCE for {TEMP_PLACED_DURATION}s")
            else:
                # predicted is not 'normal' -> cancel any picked->normal timer
                if picked_normal_start_time is not None:
                    picked_normal_start_time = None
        else:
            # not in picked state, ensure no stale timer
            picked_normal_start_time = None

        # -------- FSM UPDATE (unchanged) --------
        next_state = fsm_update(current_state, predicted)

        if next_state != current_state:
            state_counter += 1
            if state_counter >= MIN_STATE_FRAMES:
                # state transition occurs here
                prev_state = current_state
                current_state = next_state
                state_counter = 0

                # Reset the one-time latch when we leave 'picked' so future picked cycles can show again
                if current_state != "picked":
                    placed_once_after_picked = False
        else:
            state_counter = 0

        # last_conf corresponds to visible state (override shows placed)
        if display_override == "placed":
            last_conf = float(probs[CLASSES.index("placed")])
        else:
            last_conf = probs[CLASSES.index(current_state)]

        print(
            f"Frame {frame_idx:05d} | "
            f"Pred={predicted:7s} | "
            f"FSM={current_state:7s} | "
            f"Conf={last_conf:.2f}"
        )

        # If we were in seeking and inference ran post-buffer-fill -> stop seeking and resume display
        if seeking and len(buffer) == SEQ_LEN:
            inference_done_since_seek = True
            seeking = False
            print("[seek] Buffer refilled and first inference done — resuming FSM display")

    # --------------------------------------------------------
    # DISPLAY OUTPUT
    # --------------------------------------------------------
    visible_state = display_override if (display_override is not None and not seeking) else current_state

    # Only show FSM state when not seeking; while seeking we showed buffer progress above
    if not seeking:
        cv2.putText(
            display,
            f"STATE: {visible_state.upper()}  CONF: {last_conf:.2f}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            COLORS[visible_state],
            3
        )

    # If the picked->normal timer is running, show a small timer indicator
    if picked_normal_start_time is not None and not temp_placed_active:
        elapsed = time.time() - picked_normal_start_time
        remain = max(0.0, WAIT_BEFORE_TEMP - elapsed)
        timer_text = f"pending placed in: {remain:.2f}s"
        cv2.putText(display, timer_text, (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 50), 2)

    # Show frame index / FPS info
    info_text = f"Frame: {frame_idx}/{total_frames if total_frames>0 else '?'}  FPS: {fps:.1f}"
    cv2.putText(display, info_text, (30, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

    cv2.imshow(WINDOW_NAME, display)

    # Handle mouse clicks (button areas)
    if _mouse_click["clicked"]:
        mx, my = _mouse_click["x"], _mouse_click["y"]
        _mouse_click["clicked"] = False
        # BACK button click
        if left_btn_tl[0] <= mx <= left_btn_br[0] and left_btn_tl[1] <= my <= left_btn_br[1]:
            skip_frames = int(SKIP_SECONDS * fps)
            target = frame_idx - skip_frames
            new_idx = jump_frames(cap, target)
            print(f"[button] BACK clicked -> jumped to frame {new_idx}")
            continue
        # FORWARD button click
        if right_btn_tl[0] <= mx <= right_btn_br[0] and right_btn_tl[1] <= my <= right_btn_br[1]:
            skip_frames = int(SKIP_SECONDS * fps)
            target = frame_idx + skip_frames
            new_idx = jump_frames(cap, target)
            print(f"[button] FORWARD clicked -> jumped to frame {new_idx}")
            continue

    # Keyboard handling
    key = cv2.waitKey(1) & 0xFF
    if key != 255:
        if key == ord("q"):
            break
        if key == ord("f"):
            skip_frames = int(SKIP_SECONDS * fps)
            target = frame_idx + skip_frames
            new_idx = jump_frames(cap, target)
            print(f"[key] 'f' pressed -> jumped to frame {new_idx}")
            continue
        if key == ord("b"):
            skip_frames = int(SKIP_SECONDS * fps)
            target = frame_idx - skip_frames
            new_idx = jump_frames(cap, target)
            print(f"[key] 'b' pressed -> jumped to frame {new_idx}")
            continue
        # optional: step frame-by-frame with 'n' (next) and 'p' (prev)
        if key == ord("n"):
            new_idx = jump_frames(cap, frame_idx + 1)
            print(f"[key] 'n' pressed -> stepped to frame {new_idx}")
            continue
        if key == ord("p"):
            new_idx = jump_frames(cap, frame_idx - 1)
            print(f"[key] 'p' pressed -> stepped to frame {new_idx}")
            continue

# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()
print("✅ Finished")
