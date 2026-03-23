"""
split_dataset_videos_safe.py

Splits a dataset of class-subfolders containing videos into train/val,
renaming destination files when a filename already exists so nothing is lost.

Usage:
    python split_dataset_videos_safe.py
"""

import os
import random
import shutil
import csv
from pathlib import Path

# ================= CONFIG =================
SOURCE_ROOT = r"E:\kalmardatavideoscam302" # where extracted videos exist
OUTPUT_ROOT = r"E:\kalmar_dataset_split21"

TRAIN_RATIO = 0.7
# VAL_RATIO not needed (computed as remainder); we use rounding to compute counts.
RANDOM_SEED = 42
MOVE_INSTEAD_OF_COPY = False  # If True, files will be moved instead of copied

# Allowed video extensions (lowercase)
ALLOWED_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")

CLASSES = ["normal", "picked", "placed"]
# =========================================

random.seed(RANDOM_SEED)

def ensure_dirs():
    for split in ("train", "val"):
        for cls in CLASSES:
            p = os.path.join(OUTPUT_ROOT, split, cls)
            os.makedirs(p, exist_ok=True)

def collect_videos():
    """
    Collect video paths per class.
    Returns: dict {cls: [abs_path, ...]}
    """
    data = {cls: [] for cls in CLASSES}

    for entry in os.listdir(SOURCE_ROOT):
        video_folder_path = os.path.join(SOURCE_ROOT, entry)
        if not os.path.isdir(video_folder_path):
            continue

        # look for class subfolders inside this folder (as your original layout)
        for cls in CLASSES:
            cls_path = os.path.join(video_folder_path, cls)
            if not os.path.isdir(cls_path):
                continue
            for fname in os.listdir(cls_path):
                if not fname:
                    continue
                if os.path.splitext(fname)[1].lower() in ALLOWED_EXTS:
                    data[cls].append(os.path.join(cls_path, fname))

    return data

def unique_destination_path(dst_dir, filename):
    """
    If filename already exists in dst_dir, append _1, _2, ... before extension.
    Returns full path (not creating file).
    """
    base, ext = os.path.splitext(filename)
    candidate = os.path.join(dst_dir, filename)
    counter = 1
    while os.path.exists(candidate):
        candidate = os.path.join(dst_dir, f"{base}_{counter}{ext}")
        counter += 1
    return candidate

def safe_copy_move(src_path, dst_dir):
    """
    Copy (or move) src_path into dst_dir ensuring no overwrite by renaming if needed.
    Returns the actual destination path used.
    """
    os.makedirs(dst_dir, exist_ok=True)
    filename = os.path.basename(src_path)
    dst = unique_destination_path(dst_dir, filename)

    if MOVE_INSTEAD_OF_COPY:
        shutil.move(src_path, dst)
    else:
        shutil.copy2(src_path, dst)

    return dst

def split_and_copy(data):
    mapping_rows = []  # for CSV: original, split, class, dst_path
    summary = {}

    for cls, videos in data.items():
        random.shuffle(videos)
        n_total = len(videos)
        if n_total == 0:
            summary[cls] = (0, 0, 0)
            print(f"[{cls}] total=0 -> skipping")
            continue

        n_train = round(n_total * TRAIN_RATIO)
        # ensure sum equals total
        n_val = n_total - n_train

        train_videos = videos[:n_train]
        val_videos = videos[n_train:]

        print(f"[{cls}] total={n_total}, train={len(train_videos)}, val={len(val_videos)}")

        # copy/move
        for v in train_videos:
            dst_dir = os.path.join(OUTPUT_ROOT, "train", cls)
            dst_path = safe_copy_move(v, dst_dir)
            mapping_rows.append((v, "train", cls, dst_path))

        for v in val_videos:
            dst_dir = os.path.join(OUTPUT_ROOT, "val", cls)
            dst_path = safe_copy_move(v, dst_dir)
            mapping_rows.append((v, "val", cls, dst_path))

        summary[cls] = (n_total, len(train_videos), len(val_videos))

    # write mapping CSV
    csv_path = os.path.join(OUTPUT_ROOT, "split_mapping.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["original_path", "split", "class", "dest_path"])
        writer.writerows(mapping_rows)

    return summary, csv_path

def verify_output_counts():
    """
    Count actual files on disk under OUTPUT_ROOT/train and OUTPUT_ROOT/val per class.
    Returns dict {(split,cls): count}
    """
    counts = {}
    for split in ("train", "val"):
        for cls in CLASSES:
            p = os.path.join(OUTPUT_ROOT, split, cls)
            if not os.path.isdir(p):
                counts[(split, cls)] = 0
                continue
            count = len([f for f in os.listdir(p) if os.path.splitext(f)[1].lower() in ALLOWED_EXTS])
            counts[(split, cls)] = count
    return counts

def main():
    print("Preparing directories...")
    ensure_dirs()
    print("Collecting videos from source...")
    data = collect_videos()
    total_videos = sum(len(v) for v in data.values())
    print(f"Found total {total_videos} videos across classes: " + ", ".join(f"{c}={len(data[c])}" for c in CLASSES))

    summary, csv_path = split_and_copy(data)
    print("\nSummary (planned):")
    for cls, (tot, tr, va) in summary.items():
        print(f"  {cls}: total={tot}, train={tr}, val={va}")

    print(f"\nMapping saved to: {csv_path}")

    print("\nVerifying actual files on disk (post-copy/move):")
    counts = verify_output_counts()
    for (split, cls), cnt in counts.items():
        print(f"  {split}/{cls}: {cnt}")

    print("\n✅ Dataset split completed")

if __name__ == "__main__":
    main()
