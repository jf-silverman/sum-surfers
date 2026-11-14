"""watch_live.collect_once in --primary-only mode (2026-11-13): one frame kept
and counted per grab, for dense all-day runs off the free live stream."""

import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import watch_live as wl  # noqa: E402


def fake_clip(path):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (1280, 720))
    for _ in range(60):
        w.write(np.full((720, 1280, 3), 128, np.uint8))
    w.release()


def test_primary_only_keeps_and_counts_one_frame(tmp_path, monkeypatch):
    src = tmp_path / "src.mp4"; fake_clip(src)
    monkeypatch.setattr(wl, "record_clip", lambda url, dest, duration_sec=6: shutil.copy(src, dest))
    monkeypatch.setattr(wl, "check_clip_dimensions", lambda p: None)
    monkeypatch.setattr(wl.ds, "compute_image_quality", lambda p: (True, "ok", 128.0, 300.0))
    monkeypatch.setattr(wl.ds, "compute_glare_frac", lambda p: 0.0)
    monkeypatch.setattr(wl.ds, "run_inference_with_boxes",
                        lambda m, p: [(0, 0, 5, 5, 0.6), (10, 10, 15, 15, 0.8)])
    monkeypatch.setattr(wl.ds, "append_row", lambda csv, row: None)
    frames = tmp_path / "frames"; frames.mkdir()
    row = wl.collect_once(object(), "url", frames, tmp_path / "out.csv", primary_only=True)
    assert row["surfer_count"] == 2 and row["frame_count_2"] == 2
    assert row["frame_count_1"] == "" and row["frame_count_3"] == ""
    assert row["confidence_avg"] == 0.7
    assert [p.suffix for p in frames.iterdir()] == [".jpg"]       # one frame, clip deleted
