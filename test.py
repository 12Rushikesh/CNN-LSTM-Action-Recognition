# ============================================================
# VIDEO TESTING – CNN + LSTM (MATCHES TRAINING ARCHITECTURE)
# WITH CLASS-SPECIFIC CONFIDENCE THRESHOLDS (OPTION 4)
# ============================================================

import cv2                     # Video processing
import torch                   # PyTorch core
import numpy as np             # Numerical ops
import torchvision.transforms as T
import torchvision.models as models
import torch.nn as nn
from collections import deque  # Sliding window buffer

# ------------------------------------------------------------
# WINDOWS + CUDA SAFETY
# Disable cuDNN to avoid LSTM crashes on Windows
# ------------------------------------------------------------
torch.backends.cudnn.enabled = False

# ============================================================
# CONFIGURATION
# ============================================================

# Path to trained model checkpoint
MODEL_PATH = r"D:\Rushikesh\project\CNN\LSTM-train\models\model2\3.pth"

# Path to test video
VIDEO_PATH = r"C:\Users\Rapportsoft\Documents\kalmardata\10.mp4"

# Class labels (must match training)
CLASSES = ["normal", "picked", "placed"]

# Sequence length used during training
SEQ_LEN = 16

# Image size used during training
IMG_SIZE = 224

# Run inference every N frames
INFER_EVERY = 8

# Device selection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------
# OPTION 4: CLASS-SPECIFIC CONFIDENCE THRESHOLDS
# ------------------------------------------------------------
CLASS_THRESHOLDS = {
    "normal": 0.85,
    "picked": 0.60,
    "placed": 0.40  # 🔥 LOWER threshold for rare class
}

# Colors for display (visual debugging)
COLORS = {
    "normal": (200, 200, 200),
    "picked": (0, 255, 0),
    "placed": (0, 165, 255)  # Orange
}

# ============================================================
# MODEL DEFINITION (EXACT SAME AS TRAINING)
# ============================================================
class CNN_LSTM_Industry(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # -------- CNN Backbone (EfficientNet-B0) --------
        backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        self.cnn = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # -------- CNN Feature Projection --------
        self.cnn_fc = nn.Sequential(
            nn.Linear(1280, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # -------- LSTM (Temporal Modeling) --------
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            dropout=0.0
        )

        # -------- Classifier --------
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x shape: (B, T, C, H, W)
        B, T, C, H, W = x.shape

        # Merge batch & time for CNN
        x = x.view(B * T, C, H, W)

        # CNN forward
        f = self.cnn(x)
        f = self.pool(f).view(f.size(0), -1)
        f = self.cnn_fc(f)

        # Restore time dimension
        f = f.view(B, T, -1)

        # LSTM forward
        out, _ = self.lstm(f)

        # Temporal average
        out = out.mean(dim=1)

        # Classification
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
# IMAGE TRANSFORM (MUST MATCH TRAINING)
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
# VIDEO INFERENCE LOOP
# ============================================================

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("❌ Cannot open video")

cv2.namedWindow("Kalmar Action Detection", cv2.WINDOW_NORMAL)

# Sliding window buffer for frames
buffer = deque(maxlen=SEQ_LEN)

frame_idx = 0
last_pred = "normal"
last_conf = 0.0

print("▶ Video processing started")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    display = frame.copy()

    # --------------------------------------------------------
    # PREPROCESS FRAME FOR MODEL
    # --------------------------------------------------------
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = transform(rgb)
    buffer.append(tensor)

    # --------------------------------------------------------
    # RUN INFERENCE EVERY N FRAMES
    # --------------------------------------------------------
    if frame_idx % INFER_EVERY == 0 and len(buffer) == SEQ_LEN:
        x = torch.stack(list(buffer)).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        # ---------------- OPTION 4 LOGIC ----------------
        best_idx = int(np.argmax(probs))
        best_class = CLASSES[best_idx]
        best_conf = float(probs[best_idx])

        final_pred = "normal"
        final_conf = best_conf

        # If predicted class meets its threshold
        if best_conf >= CLASS_THRESHOLDS[best_class]:
            final_pred = best_class
            final_conf = best_conf
        else:
            # Rescue logic for placed
            placed_idx = CLASSES.index("placed")
            placed_conf = float(probs[placed_idx])

            if placed_conf >= CLASS_THRESHOLDS["placed"]:
                final_pred = "placed"
                final_conf = placed_conf
            else:
                final_pred = "normal"
                final_conf = best_conf

        last_pred = final_pred
        last_conf = final_conf

        print(f"Frame {frame_idx}: {last_pred} ({last_conf:.2f})")

    # --------------------------------------------------------
    # DRAW OUTPUT
    # --------------------------------------------------------
    color = COLORS[last_pred]

    cv2.putText(
        display,
        f"ACTION: {last_pred.upper()}  CONF: {last_conf:.2f}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        color,
        3
    )

    cv2.imshow("Kalmar Action Detection", display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ============================================================
# CLEANUP
# ============================================================
cap.release()
cv2.destroyAllWindows()
print("✅ Video finished")
