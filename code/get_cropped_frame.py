import cv2
from pathlib import Path
from datetime import datetime
import json

# ---------- CONFIG ----------
INPUT_DIR = Path("data/not_needed_in_repo/surf_clips")           # Folder containing date/time subfolders
OUTPUT_DIR = Path("data/j_shore_cam/surf_crops")         # Where cropped frames will go
LOG_FILE = OUTPUT_DIR / "skipped_frames.log"
FRAME_TIME_SEC = 2.5                     # Take frame at 2.5s into clip
ROI_X, ROI_Y, ROI_W, ROI_H = 0, 420, 1280, 180  # Crop region
# --------------------------------


def extract_single_frame(video_path, output_path):
    """Extract one frame from a video and crop ROI."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        raise RuntimeError(f"Cannot get FPS for {video_path}")
    frame_num = int(FRAME_TIME_SEC * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret or frame is None:
        raise RuntimeError(f"Failed to read frame {frame_num} from {video_path}")

    # Crop ROI (y, y+h, x, x+w)
    crop = frame[ROI_Y:ROI_Y + ROI_H, ROI_X:ROI_X + ROI_W]

    # Save without compression (max quality)
    success = cv2.imwrite(str(output_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 100])
    cap.release()
    
    # or Save as PNG
    # output_path = output_path.with_suffix(".png")
    # success = cv2.imwrite(str(output_path), crop)

    if not success:
        raise RuntimeError(f"Failed to write frame to {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    skipped = []

    for clip_path in INPUT_DIR.rglob("clip.mp4"):
        try:
            date_folder = clip_path.parent.parent.name  # YYYY-MM-DD
            time_folder = clip_path.parent.name         # HH_MM
            dt = datetime.strptime(f"{date_folder}_{time_folder}", "%Y-%m-%d_%H_%M")
            frame_name = f"crop{dt.strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
            output_path = OUTPUT_DIR / frame_name

            if output_path.exists():
                skipped.append(frame_name)
                continue

            extract_single_frame(clip_path, output_path)
            print(f"✅ Saved frame {frame_name}")

        except Exception as e:
            print(f"⚠️ Error with {clip_path}: {e}")

    # Log skipped frames
    if skipped:
        with open(LOG_FILE, "a") as f:
            for name in skipped:
                f.write(name + "\n")
        print(f"📝 Logged {len(skipped)} skipped frames to {LOG_FILE}")


if __name__ == "__main__":
    main()