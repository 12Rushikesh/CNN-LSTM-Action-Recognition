# ============================================================
# SAFE CNN + LSTM VIDEO TRAINING (WINDOWS STABLE)
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

# ----------------- CRITICAL FIX -----------------
# Prevent cuDNN LSTM crash on Windows
torch.backends.cudnn.enabled = False
# ------------------------------------------------

# ================= CONFIG =================
DATASET_ROOT = r"E:\kalmar_dataset_split"
TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VAL_DIR   = os.path.join(DATASET_ROOT, "val")

CLASSES = ["normal", "picked", "placed"]
CLASS2ID = {c: i for i, c in enumerate(CLASSES)}

SEQ_LEN = 16
IMG_SIZE = 224
BATCH_SIZE = 4
EPOCHS = 15
LR = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_OUT = r"D:\Rushikesh\project\CNN\LSTM-train\models\model2/4.pth"
# =========================================


# ============================================================
# DATASET
# ============================================================
class VideoDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        for cls in CLASSES:
            for v in glob(os.path.join(root_dir, cls, "*.mp4")):
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

    def sample_frames(self, path):
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
            raise RuntimeError(f"Empty or broken video: {path}")

        while len(frames) < SEQ_LEN:
            frames.append(frames[-1])

        return torch.stack(frames)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.sample_frames(path), label


# ============================================================
# MODEL
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

        # -------- SAFE LSTM (NO CRASH) --------
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
    print("Class distribution (train):", counts)

    # ---- Class balancing (SAMPLER ONLY) ----
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
        num_workers=4,
        pin_memory=True
    )

    model = CNN_LSTM_Industry(len(CLASSES)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5
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

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                preds = model(x).argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        val_acc = correct / total
        scheduler.step(val_acc)

        print(f"Epoch {epoch:02d} | Train {train_acc:.3f} | Val {val_acc:.3f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(
                {"model": model.state_dict(), "classes": CLASSES},
                MODEL_OUT
            )
            print("✅ Saved best model")

    print("\nTraining complete.")
    print("Best Validation Accuracy:", best_val)


if __name__ == "__main__":
    train()
