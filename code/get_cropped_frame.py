import cv2
from pathlib import Path
from datetime import datetime
import json

# ---------- CONFIG ----------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = _PROJECT_ROOT / "data" / "not_needed_in_repo" / "surf_clips"
OUTPUT_DIR = _PROJECT_ROOT / "data" / "j_shore_cam" / "surf_crops"
LOG_FILE = OUTPUT_DIR / "skipped_frames.log"
FRAME_TIME_SEC = 2.5                     # Primary frame, same as always — unsuffixed
                                          # filename (crop<ts>.jpg). Everything built
                                          # around this single image (quality-gate
                                          # thresholds, review batches, etc.) keeps
                                          # working unchanged.
SIDE_FRAME_OFFSET_SEC = 1.5              # Two extra frames at FRAME_TIME_SEC +/- this,
                                          # for multi-frame count averaging in
                                          # detect_surfers.py (see docs/PROJECT_HISTORY.md,
                                          # 2026-08-25 entry — per-frame counts on the
                                          # same clip vary by several surfers within a
                                          # few seconds, so averaging reduces noise).
ROI_X, ROI_Y, ROI_W, ROI_H = 0, 420, 1280, 180  # Crop region
# --------------------------------


def extract_frame_at(video_path, time_sec, output_path):
    """Extract one frame at `time_sec` into the clip and crop ROI."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        raise RuntimeError(f"Cannot get FPS for {video_path}")
    frame_num = int(time_sec * fps)

    # Deliberately NOT using cap.set(CAP_PROP_POS_FRAMES, ...) to seek — on these
    # clips it's unreliable in a way that isn't just "fails on far frames": a
    # fresh capture can fail to seek to a mid-stream frame at all, and adding a
    # single warm-up read (which fixes seeking to LATER frames) then breaks
    # seeking to EARLIER/low frame numbers instead (position reads back as a
    # nonsensical -5.0/-4.0, and it silently re-returns the warm-up frame).
    # Sequential .read() is reliable for every frame in every clip tested
    # (confirmed empirically) and these clips are short (~150 frames), so
    # reading forward to the target frame is the robust choice here, not a
    # performance-motivated shortcut.
    frame = None
    for i in range(frame_num + 1):
        ret, candidate = cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame {frame_num} from {video_path} "
                                f"(clip ended at frame {i})")
        frame = candidate

    # Crop ROI (y, y+h, x, x+w)
    crop = frame[ROI_Y:ROI_Y + ROI_H, ROI_X:ROI_X + ROI_W]

    # Save without compression (max quality)
    success = cv2.imwrite(str(output_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 100])
    cap.release()

    if not success:
        raise RuntimeError(f"Failed to write frame to {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    skipped = []

    side_times = [FRAME_TIME_SEC - SIDE_FRAME_OFFSET_SEC, FRAME_TIME_SEC + SIDE_FRAME_OFFSET_SEC]

    for clip_path in INPUT_DIR.rglob("clip.mp4"):
        try:
            date_folder = clip_path.parent.parent.name  # YYYY-MM-DD
            time_folder = clip_path.parent.name         # HH_MM
            dt = datetime.strptime(f"{date_folder}_{time_folder}", "%Y-%m-%d_%H_%M")
            base_name = f"crop{dt.strftime('%Y-%m-%d_%H-%M-%S')}"

            # Primary frame (unsuffixed) — unchanged behavior.
            frame_name = f"{base_name}.jpg"
            output_path = OUTPUT_DIR / frame_name
            if output_path.exists():
                skipped.append(frame_name)
            else:
                extract_frame_at(clip_path, FRAME_TIME_SEC, output_path)
                print(f"✅ Saved frame {frame_name}")

            # Side frames for multi-frame count averaging.
            for i, t in enumerate(side_times, start=1):
                side_name = f"{base_name}_side{i}.jpg"
                side_path = OUTPUT_DIR / side_name
                if side_path.exists():
                    skipped.append(side_name)
                    continue
                extract_frame_at(clip_path, t, side_path)
                print(f"✅ Saved frame {side_name}")

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
