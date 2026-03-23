# ============================================================
# KALMAR ACTION RECOGNITION — FINE-TUNING TRAINING
# CNN + LSTM (EfficientNet-B0 backbone)
# Classes: normal, picked
# Includes:
# - Pretrained model loading (fine-tuning)
# - Class imbalance handling
# - Confusion matrix
# - Windows-safe LSTM
# ============================================================

import os
import cv2
from glob import glob
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.models as models
import torchvision.transforms as T

import numpy as np
from sklearn.metrics import confusion_matrix

# ---------------- WINDOWS SAFETY ----------------
torch.backends.cudnn.enabled = False
# ------------------------------------------------

# ================= CONFIG =================
DATASET_ROOT = r"F:\AI projects\kalmar_dataset_split_2sec_clips_CAM1"
TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VAL_DIR   = os.path.join(DATASET_ROOT, "val")

CLASSES = ["normal", "picked"]
CLASS2ID = {c: i for i, c in enumerate(CLASSES)}

SEQ_LEN = 16
IMG_SIZE = 224
BATCH_SIZE = 4
EPOCHS = 6

# 🔽 LOWER LR FOR FINE-TUNING
LR = 1e-5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 🔁 PRETRAINED MODEL (FROM YOUR PREVIOUS TRAINING)
PRETRAINED_MODEL = r"D:\Rushikesh\project\CNN\LSTM-train\models\kalmar_dataset_split_2sec_clips_CAM1\useinlive\kml2.pth"

# 💾 SAVE NEW FINE-TUNED MODEL
MODEL_OUT = r"D:\Rushikesh\project\CNN\LSTM-train\models\kalmar_dataset_split_2sec_clips_CAM1\Re\kml_finetune_2.pth"
# =========================================


# ============================================================
# DATASET
# ============================================================
class VideoDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []

        for cls in CLASSES:
            cls_dir = os.path.join(root_dir, cls)
            for v in glob(os.path.join(cls_dir, "*.mp4")):
                self.samples.append((v, CLASS2ID[cls]))

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.samples)

    def _load_frames(self, path):
        cap = cv2.VideoCapture(path)
        frames = []

        while len(frames) < SEQ_LEN:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(self.transform(frame))

        cap.release()

        if len(frames) == 0:
            raise RuntimeError(f"Broken video: {path}")

        while len(frames) < SEQ_LEN:
            frames.append(frames[-1])

        return torch.stack(frames)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self._load_frames(path), label


# ============================================================
# MODEL (UNCHANGED ARCHITECTURE)
# ============================================================
class CNN_LSTM_Industry(nn.Module):
    def __init__(self):
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
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, len(CLASSES))
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
# TRAINING
# ============================================================
def train():
    train_ds = VideoDataset(TRAIN_DIR)
    val_ds   = VideoDataset(VAL_DIR)

    labels = [y for _, y in train_ds.samples]
    counts = Counter(labels)
    print("📊 Train class distribution:", counts)

    weights = [1.0 / counts[y] for y in labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )

    # -------- MODEL --------
    model = CNN_LSTM_Industry().to(DEVICE)

    print("🔁 Loading pretrained weights for fine-tuning...")
    ckpt = torch.load(PRETRAINED_MODEL, map_location=DEVICE)
    model.load_state_dict(ckpt["model"], strict=True)
    print("✅ Pretrained model loaded")

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )

    best_val = 0.0

    for epoch in range(1, EPOCHS + 1):
        # -------- TRAIN --------
        model.train()
        correct, total = 0, 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_acc = correct / total

        # -------- VALID --------
        model.eval()
        correct, total = 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                preds = logits.argmax(1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

                correct += (preds == y).sum().item()
                total += y.size(0)

        val_acc = correct / total
        cm = confusion_matrix(all_labels, all_preds)

        print(f"\nEpoch {epoch:02d}")
        print(f"Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")
        print("Confusion Matrix:")
        print(cm)

        if val_acc > best_val:
            best_val = val_acc
            torch.save(
                {"model": model.state_dict(), "classes": CLASSES},
                MODEL_OUT
            )
            print("✅ Saved best fine-tuned model")

    print("\n🎯 Training complete")
    print("Best Validation Accuracy:", best_val)


if __name__ == "__main__":
    train()
