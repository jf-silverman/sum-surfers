"""
track_frames.py
---------------
Follows detected surfers across a clip's frames and counts who is present at
one moment, including surfers the detector missed in that frame but caught in
the frames around it (added 2026-11-16).

Why: on Joel's batch 8, where he boxed every surfer by scrubbing the clip, the
single-frame detector counted 0.86 of the truth (mostly visible surfers in foam
or partly behind a wave, missed in that frame); counting surfers present at the
frame's moment across +/-2 s brought it to 0.98, and against the 215 hand
counts it scored MAE 1.18 against 1.33 for the median/max hybrid. Recorded as
predictions.csv `frame_count_tracked` beside the hybrid; not the official count
yet (December decision).

Method: Hungarian assignment (scipy linear_sum_assignment) on box-centre
distance between consecutive frames, gated by a speed allowance (faster for
tall boxes, which are mostly standing riders), tracks allowed to skip frames,
then broken tracks re-linked when the implied speed is plausible. A surfer is
present at time t if its track has a detection at t, or detections both before
and after t. The detection-association step of SORT-style trackers (Bewley et
al. 2016) without the motion model: over a few seconds sitters and paddlers
barely move.

Mirrors analysis/occlusion_tracking/track_union.py, where it was developed.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

GATE_FLOOR_PX = 6.0
GATE_PX_PER_SEC = 10.0
MAX_SKIP_SEC = 3.0
TALL_PX = 17                 # standing riders are mostly 17+ px tall; sitting/prone median 10
TALL_PX_PER_SEC = 40.0
LINK_MAX_GAP_SEC = 2.0


def _centres(boxes):
    return np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in boxes]).reshape(-1, 2)


def _height(b):
    return b[3] - b[1]


def track(frames):
    """frames: [(t_sec, boxes)] in time order, boxes [x1, y1, x2, y2, ...].
    Returns tracks, each a list of (t, box)."""
    tracks = []
    for t, boxes in frames:
        live = [k for k, tr in enumerate(tracks) if t - tr[-1][0] <= MAX_SKIP_SEC]
        c = _centres(boxes)
        assigned = set()
        if live and len(boxes):
            last = np.array([_centres([tracks[k][-1][1]])[0] for k in live])
            gaps = np.array([t - tracks[k][-1][0] for k in live])
            speed = np.array([TALL_PX_PER_SEC if _height(tracks[k][-1][1]) >= TALL_PX else GATE_PX_PER_SEC
                              for k in live])
            d = np.linalg.norm(last[:, None, :] - c[None, :, :], axis=2)
            cost = np.where(d <= (GATE_FLOOR_PX + speed * gaps)[:, None], d, 1e6)
            for r, col in zip(*linear_sum_assignment(cost)):
                if cost[r, col] < 1e6:
                    tracks[live[r]].append((t, boxes[col]))
                    assigned.add(col)
        tracks.extend([(t, b)] for j, b in enumerate(boxes) if j not in assigned)
    return _link(tracks)


def _link(tracks):
    """Join a track that ends to one starting shortly after, nearby, never
    overlapping in time (one surfer cannot be two boxes in one frame)."""
    tracks = sorted(tracks, key=lambda tr: tr[0][0])
    merged = True
    while merged:
        merged = False
        for i, a in enumerate(tracks):
            end_t, end_b = a[-1]
            best, best_d = None, None
            for j, b in enumerate(tracks):
                if j == i or b[0][0] <= end_t or b[0][0] - end_t > LINK_MAX_GAP_SEC:
                    continue
                gap = b[0][0] - end_t
                speed = TALL_PX_PER_SEC if max(_height(end_b), _height(b[0][1])) >= TALL_PX else GATE_PX_PER_SEC
                d = float(np.linalg.norm(_centres([end_b])[0] - _centres([b[0][1]])[0]))
                if d <= GATE_FLOOR_PX + speed * gap and (best_d is None or d < best_d):
                    best, best_d = j, d
            if best is not None:
                tracks[i] = a + tracks[best]
                del tracks[best]
                merged = True
                break
    return tracks


def present_count(frames, t0):
    """Surfers present at t0: a track with a detection at t0, or detections
    both before and after it."""
    n = 0
    for tr in track(frames):
        ts = [t for t, _ in tr]
        n += (t0 in ts) or (min(ts) < t0 < max(ts))
    return n
