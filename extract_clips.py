
# Extract EAch Video lenth clips 2 seconds with 1 second step for cnn-lstm model training
import os
import cv2
from collections import deque

# ================= CONFIG =================
INPUT_ROOT = r"E:\kalmar_dataset_split21"
OUTPUT_ROOT = r"E:\CAM6"

SPLITS = ["train", "val"]
CLASSES = ["normal", "picked"]

CLIP_DURATION_SEC = 2   # length of each clip
STEP_SEC = 1            # sliding window step

VIDEO_EXTS = (".mp4", ".avi", ".mov")
# =========================================


def extract_clips(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        cap.release()
        print(f"⚠️ Skipping (FPS error): {video_path}")
        return

    clip_frames = int(CLIP_DURATION_SEC * fps)
    step_frames = int(STEP_SEC * fps)

    buffer = deque(maxlen=clip_frames)
    frame_idx = 0
    clip_idx = 0

    video_name = os.path.splitext(os.path.basename(video_path))[0]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        buffer.append(frame)
        frame_idx += 1

        # create clip every STEP_SEC
        if len(buffer) == clip_frames and frame_idx % step_frames == 0:
            h, w, _ = frame.shape

            out_path = os.path.join(
                output_dir,
                f"{video_name}_clip{clip_idx:03d}.mp4"
            )

            writer = cv2.VideoWriter(
                out_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (w, h)
            )

            for f in buffer:
                writer.write(f)

            writer.release()
            clip_idx += 1

    cap.release()


def process_dataset():
    total_videos = 0
    total_clips = 0

    for split in SPLITS:
        for cls in CLASSES:
            input_dir = os.path.join(INPUT_ROOT, split, cls)
            output_dir = os.path.join(OUTPUT_ROOT, split, cls)

            if not os.path.exists(input_dir):
                print(f"❌ Missing folder: {input_dir}")
                continue

            os.makedirs(output_dir, exist_ok=True)

            videos = [
                f for f in os.listdir(input_dir)
                if f.lower().endswith(VIDEO_EXTS)
            ]

            print(f"\n📂 Processing {split}/{cls} | Videos: {len(videos)}")

            for file in videos:
                extract_clips(
                    os.path.join(input_dir, file),
                    output_dir
                )
                total_videos += 1

            clip_count = len(os.listdir(output_dir))
            total_clips += clip_count

            print(f"✅ Done {split}/{cls} | Clips: {clip_count}")

    print("\n===================================")
    print(f"🎉 EXTRACTION COMPLETE")
    print(f"📽️ Total videos processed: {total_videos}")
    print(f"🎞️ Total clips generated: {total_clips}")
    print("===================================")


if __name__ == "__main__":
    process_dataset()
