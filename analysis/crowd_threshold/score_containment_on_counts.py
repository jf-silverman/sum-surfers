"""Does loosening containment suppression in crowds survive the sparse frames?

Box-level truth says `CONTAINMENT_THRESH = 0.7` costs more real surfers than the
duplicates it removes: on the 61 held-out labelled frames only 10 of 183 false
positives are a second box on an already-counted surfer, while loosening the rule
in dense neighbourhoods lifts recall and cuts count error monotonically.

That set cannot be the whole answer, because containment suppression was adopted
on 2026-09-22 for a failure the labelled frames do not contain: clear-day
OVERcount on near-empty water (60-second clear-day counts 107.1% -> 102.0%,
MAE 1.90 -> 1.17). The labelled set's emptiest quartile still averages 8.3
surfers. Relaxing a rule on crowded evidence and shipping it into sparse frames
is the same mistake the flat 0.150 threshold made.

So this re-measures every containment variant on the 263 hand-counted frames,
which do contain the sparse end (emptiest quartile: mean 1.1), at count level
against the hand counts -- the same frames, truth and bootstrap as
`analysis/conf_thresh_recheck/`.

The density-adaptive variant is the one to watch. Its threshold is
`0.7 + slope * neighbours`, so a box with NO neighbour within 40px keeps exactly
production's 0.7: by construction it cannot touch an isolated clear-day false
positive, which is what turning the rule off would do. If it holds the sparse
frames and gains on the crowded ones, it is the instrument; if it loses the
sparse frames too, containment stays flat.

Caches floored proposals on first run so later sweeps need no inference.

Usage:
    python analysis/crowd_threshold/score_containment_on_counts.py
    python analysis/crowd_threshold/score_containment_on_counts.py --rebuild
"""
from __future__ import annotations

import argparse
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "code"))
sys.path.insert(0, str(_ROOT / "analysis" / "conf_thresh_recheck"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import detect_surfers as ds  # noqa: E402
from recheck_threshold import load_spotcheck, load_review_days, FLOOR  # noqa: E402
from score_duplicate_rules import suppress_adaptive  # noqa: E402

HERE = Path(__file__).resolve().parent
CROPS = _ROOT / "data" / "j_shore_cam" / "surf_crops"
CACHE = HERE / "handcount_proposals.pkl"
PROD_THRESH = 0.195
N_BOOT = 5000
SEED = 17

VARIANTS = [
    ("flat 0.7 (production)", 0.7, 0.0),
    ("looser in crowds  0.7 +0.05/nb", 0.7, 0.05),
    ("looser in crowds  0.7 +0.10/nb", 0.7, 0.10),
    ("looser in crowds  0.7 +0.15/nb", 0.7, 0.15),
    ("flat 0.9", 0.9, 0.0),
    ("off", 1.01, 0.0),
    ("tighter in crowds 0.7 -0.05/nb", 0.7, -0.05),
]


def build_cache():
    frames = pd.concat([load_spotcheck(), load_review_days()], ignore_index=True)
    frames = frames[[(CROPS / f).exists() for f in frames.filename]].reset_index(drop=True)
    print(f"{len(frames)} hand-counted frames with a crop on disk")
    model = ds.load_model()
    store = {}
    for i, r in enumerate(frames.itertuples(), 1):
        tiles, tmp = ds.tile_image_paths(CROPS / r.filename)
        out = []
        for _, xoff, tp in tiles:
            res = model.predict(source=str(tp), conf=FLOOR, device=ds.DEVICE,
                                verbose=False)[0]
            for b in res.boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().tolist()
                out.append([x1 + xoff, y1, x2 + xoff, y2, float(b.conf[0].cpu())])
        shutil.rmtree(tmp, ignore_errors=True)
        store[r.filename] = out
        if i % 25 == 0:
            print(f"  {i}/{len(frames)}", flush=True)
    with CACHE.open("wb") as fh:
        pickle.dump({"frames": frames, "boxes": store}, fh)
    print(f"  -> {CACHE}")
    return frames, store


def count(boxes, base, slope):
    kept = [b for b in boxes if b[4] >= PROD_THRESH]
    kept = ds.nms_across_tiles(kept)
    kept = (ds.suppress_contained(kept, base) if slope == 0
            else suppress_adaptive(kept, base, slope))
    return len(ds.filter_false_positive_zones(kept))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    if args.rebuild or not CACHE.exists():
        frames, store = build_cache()
    else:
        with CACHE.open("rb") as fh:
            c = pickle.load(fh)
        frames, store = c["frames"], c["boxes"]
        print(f"{len(frames)} frames from cache {CACHE.name}")

    d = frames.copy()
    for name, base, slope in VARIANTS:
        d[name] = [count(store[f], base, slope) for f in d.filename]

    rng = np.random.default_rng(SEED)
    prod = VARIANTS[0][0]
    rows = []
    for sname, g in d.groupby("set"):
        base_err = g[prod] - g.truth
        print(f"\n=== {sname}: {len(g)} frames, mean hand count {g.truth.mean():.1f} ===")
        print("  variant                            signed   |err|   ratio   "
              "d|err|      95% CI            p")
        for name, _, _ in VARIANTS:
            e = g[name] - g.truth
            dl = (e.abs() - base_err.abs()).to_numpy(float)
            bs = np.array([dl[rng.integers(0, len(dl), len(dl))].mean()
                           for _ in range(N_BOOT)])
            p = (np.abs(bs - dl.mean()) >= abs(dl.mean())).mean() if dl.mean() else 1.0
            print(f"  {name:<34} {e.mean():+6.2f}  {e.abs().mean():5.2f}  "
                  f"{g[name].sum() / g.truth.sum():.3f}  {dl.mean():+7.3f}  "
                  f"{np.percentile(bs, 2.5):+.3f} to {np.percentile(bs, 97.5):+.3f}  {p:.3f}")
            rows.append(dict(set=sname, variant=name, signed=e.mean(),
                             mae=e.abs().mean(), d_mae=dl.mean(), p=p))

    print("\n=== Mean |count error| by crowd quartile, pooled ===")
    d["q"] = pd.qcut(d.truth, 4, labels=["Q1 emptiest", "Q2", "Q3", "Q4 busiest"],
                     duplicates="drop")
    tab = []
    for q, sub in d.groupby("q", observed=True):
        r = dict(q=q, frames=len(sub), truth=round(sub.truth.mean(), 1))
        for name, _, _ in VARIANTS:
            r[name] = round((sub[name] - sub.truth).abs().mean(), 3)
        tab.append(r)
    print(pd.DataFrame(tab).set_index("q").T.to_string())

    pd.DataFrame(rows).to_csv(HERE / "containment_on_counts.csv", index=False)
    d.to_csv(HERE / "containment_counts_per_frame.csv", index=False)
    print(f"\n-> {HERE / 'containment_on_counts.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
