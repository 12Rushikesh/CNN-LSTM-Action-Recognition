import cv2
import torch
import numpy as np
import torchvision.transforms as T
import torchvision.models as models
import torch.nn as nn
from collections import deque

# ================= CONFIG =================
MODEL_PATH = r"D:\Rushikesh\project\CNN\LSTM-train\models\kalmar_cnn_lstm_industry.pth"
VIDEO_PATH = r"C:\Users\Rapportsoft\Documents\kalmardata\5.mkv"   # <-- CHANGE THIS

CLASSES = ["normal", "picked", "placed"]

SEQ_LEN = 16
IMG_SIZE = 224
INFER_EVERY = 8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# =========================================


# ================= MODEL =================
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
            nn.ReLU()
        )

        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=384,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Linear(768, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        f = self.cnn(x)
        f = self.pool(f).view(f.size(0), -1)
        f = self.cnn_fc(f)

        f = f.view(B, T, -1)
        out, _ = self.lstm(f)
        out = out.mean(dim=1)

        return self.softmax(self.classifier(out))


# ================= LOAD MODEL =================
print("🔹 Loading model...")
model = CNN_LSTM_Industry(len(CLASSES))
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model"], strict=False)
model.to(DEVICE)
model.eval()
print("✅ Model loaded")


# ================= TRANSFORM (MODEL ONLY) =================
transform = T.Compose([
    T.ToPILImage(),
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ================= VIDEO LOOP =================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ Cannot open video")
    exit()

# Force correct window size (NO ZOOM)
cv2.namedWindow("Kalmar Action Detection", cv2.WINDOW_NORMAL)

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

    # =============================
    # 1️⃣ DISPLAY FRAME (ORIGINAL)
    # =============================
    display_frame = frame.copy()

    # =============================
    # 2️⃣ MODEL FRAME (PREPROCESS)
    # =============================
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    model_tensor = transform(rgb)
    buffer.append(model_tensor)

    # =============================
    # 3️⃣ INFERENCE
    # =============================
    if frame_idx % INFER_EVERY == 0 and len(buffer) == SEQ_LEN:
        x = torch.stack(list(buffer)).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            probs = model(x)[0].cpu().numpy()

        idx = int(np.argmax(probs))
        last_pred = CLASSES[idx]
        last_conf = probs[idx]

        print(f"Frame {frame_idx}: {last_pred} ({last_conf:.2f})")

    # =============================
    # 4️⃣ DRAW PREDICTION
    # =============================
    cv2.putText(
        display_frame,
        f"ACTION: {last_pred.upper()}  CONF: {last_conf:.2f}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    # =============================
    # 5️⃣ SHOW (NO ZOOM)
    # =============================
    cv2.imshow("Kalmar Action Detection", display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ================= CLEANUP =================
cap.release()
cv2.destroyAllWindows()
print("✅ Video finished")
