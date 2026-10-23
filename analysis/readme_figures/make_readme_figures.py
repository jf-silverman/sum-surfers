"""
make_readme_figures.py
----------------------
The static figures the README's walk-through uses, drawn to the same style as
the nightly charts (black surface, white ink, #333333 grid, AQUA data / LIME
reference) so the page reads as one system.

These are snapshots, not nightly products: re-run this script when the record
has grown enough to change what they show. Every number printed on a figure is
computed here from the project's own records.

  pipeline_1_watch.png ... pipeline_5_forecast.png
                         one real hour followed from camera frame to forecast,
                         one image per step
  tide_vs_count.png      median count per 1-ft tide bin
  weekend_by_month.png   weekend vs weekday by month, with the ratio
  surfer_poses.png       standing vs sitting vs lying, from hand labels
  when_to_go.png         typical count by hour of day and day of week, tide < 3 ft
  seasons.png            monthly typical count and share of empty hours
  forecast_within_n.png  how often the forecast lands within N surfers
  crowd_vs_rating.png    count by Surfline's surf rating, low-to-mid tide only

Usage:
    python analysis/readme_figures/make_readme_figures.py [--only NAME ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

OUT = ROOT / "docs" / "images"
FEATURES = ROOT / "data" / "training_features.csv"
FLOG = ROOT / "data" / "forecasts" / "forecast_log.csv"
CROPS = ROOT / "data" / "j_shore_cam" / "surf_crops"
CVAT = ROOT / "data" / "cvat_out_coco"
CLIPS = ROOT / "data" / "not_needed_in_repo" / "surf_clips"

BG, INK, INK_DIM, GRID = "black", "white", "#bbbbbb", "#333333"
AQUA, LIME = "#3ab4c9", "#9de35a"
NODATA = "#1a1a1a"

# The hour followed through the walk-through. Chosen 2026-10-22 for a clear,
# high-contrast frame holding riders, sitting and lying-down surfers at once,
# on a day whose forecast range sits off zero (on 10-13 to 10-20 the lower
# edge was pinned at zero all day, which the hurdle band since fixed). The
# day's average miss was 3.9 surfers; this hour's was 10, inside the range.
PIPE_DATE, PIPE_TIME, PIPE_TIME_LABEL = "2026-09-25", "16:20", "4:20 pm"
PIPE_ZOOM = (640, 1280)     # the right half of the strip, where the lineup sits
DIM_OUTSIDE = 0.65          # brightness kept outside the counted band in step 1
# A labeled frame holding a clear example of each pose (batch01_73). The
# standing example is picked by its box's left edge: the largest "standing"
# box in the old pick was a paddleboarder, and this one is riding a wave.
POSE_FRAME = "crop2025-10-15_11-50-00.jpg"
POSE_SET = "batch01_73"
POSE_STANDING_X = 884


def _style(ax):
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.tick_params(colors=INK_DIM, labelsize=9)


def _bare(ax):
    ax.set_facecolor(BG)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(GRID)


def _save(fig, name):
    p = OUT / name
    fig.savefig(p, facecolor=BG, dpi=150)
    plt.close(fig)
    print(f"  wrote {p.relative_to(ROOT)}")


def _rgb(path):
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"cannot read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _daylight():
    t = pd.read_csv(FEATURES)
    return t[~t.is_night.astype(bool)].copy()


# --------------------------------------------------------------------------
# The walk-through: one real hour followed through the five steps, one image
# per step so the README can put each step's sentence directly above it.
def _pipe_inputs():
    fname = f"crop{PIPE_DATE}_{PIPE_TIME.replace(':', '-')}-00.jpg"
    feats = pd.read_csv(FEATURES)
    row = feats[feats.filename == fname].iloc[0]
    return fname, row


def pipe_watch():
    """The full camera frame, with the band the project keeps marked. The crop
    keeps the whole width and cuts away the sky above and the near water below
    (get_cropped_frame.ROI_*), so this shows exactly that."""
    import get_cropped_frame as gcf
    clip = CLIPS / PIPE_DATE / PIPE_TIME.replace(":", "_") / "clip.mp4"
    cap = cv2.VideoCapture(str(clip))
    cap.set(cv2.CAP_PROP_POS_MSEC, gcf.FRAME_TIME_SEC * 1000)
    ok, frame = cap.read(); cap.release()
    if not ok:
        raise SystemExit(f"cannot read a frame from {clip}")
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    y0, y1 = gcf.ROI_Y, gcf.ROI_Y + gcf.ROI_H
    dim = img.copy()
    dim[:y0] = (dim[:y0] * DIM_OUTSIDE).astype(np.uint8)
    dim[y1:] = (dim[y1:] * DIM_OUTSIDE).astype(np.uint8)

    fig = plt.figure(figsize=(11, 11 * h / w), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(dim); _bare(ax)
    ax.add_patch(Rectangle((1, y0), w - 3, y1 - y0, fill=False, edgecolor=INK,
                           linewidth=1.6, linestyle=(0, (5, 3))))
    ax.text(12, y0 - 12, f"kept for counting: the full {gcf.ROI_W} px width, "
            f"{gcf.ROI_H} of {h} px of height", color=INK, fontsize=11,
            va="bottom", fontweight="bold")
    _save(fig, "pipeline_1_watch.png")
    return f"{w}x{h}, band y {y0}-{y1}"


def pipe_spot():
    import detect_surfers as ds
    fname, _ = _pipe_inputs()
    img = _rgb(CROPS / fname)
    boxes = ds.run_inference_with_boxes(ds.load_model(), CROPS / fname)
    zx0, zx1 = PIPE_ZOOM
    sub = img[:, zx0:zx1]
    fig = plt.figure(figsize=(11, 11 * sub.shape[0] / sub.shape[1]), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(sub, interpolation="lanczos"); _bare(ax)
    n_in = 0
    for x1, y1, x2, y2, _c in boxes:
        if x2 > zx0 and x1 < zx1:
            n_in += 1
            ax.add_patch(Rectangle((x1 - zx0 - 2, y1 - 2), x2 - x1 + 4, y2 - y1 + 4,
                                   fill=False, edgecolor=LIME, linewidth=1.4))
    _save(fig, "pipeline_2_spot.png")
    return len(boxes), n_in


def pipe_count():
    _, row = _pipe_inputs()
    t = pd.Timestamp(f"{PIPE_DATE} {PIPE_TIME}")
    # 11 in wide, like steps 1 and 2, so all five render at one width
    fig = plt.figure(figsize=(11, 2.2), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1]); _bare(ax)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(0.03, 0.5, f"{row.surfer_count:g}", color=AQUA, fontsize=72,
            fontweight="bold", ha="left", va="center", transform=ax.transAxes)
    ax.text(0.22, 0.5, f"surfers at {t:%-I:%M %p}\non {t:%A, %B %-d, %Y}\n"
            "(average of three frames)", color=INK, fontsize=14,
            ha="left", va="center", transform=ax.transAxes, linespacing=1.6)
    _save(fig, "pipeline_3_count.png")
    return row.surfer_count


# Small vector icons, drawn in a unit box, so step 4 needs no icon font.
def _icon(ax, kind, color=AQUA):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")
    lw = 2.2
    x = np.linspace(0.05, 0.95, 60)
    if kind == "tide":          # water line with an up/down arrow
        ax.plot(x, 0.22 + 0.05 * np.sin(x * 4 * np.pi), color=color, lw=lw)
        ax.annotate("", (0.5, 0.95), (0.5, 0.40),
                    arrowprops=dict(arrowstyle="<|-|>", color=color, lw=lw))
    elif kind == "surf":        # a breaking wave
        th = np.linspace(0, 1.6 * np.pi, 80)
        r = 0.30 * (1 - th / (2.6 * np.pi))
        ax.plot(0.55 + r * np.cos(th + np.pi), 0.50 + r * np.sin(th + np.pi),
                color=color, lw=lw)
        ax.plot([0.05, 0.25], [0.50, 0.50], color=color, lw=lw)
        ax.plot([0.25, 0.95], [0.20, 0.20], color=color, lw=lw)
    elif kind == "swell":       # three stacked swell lines
        for y in (0.25, 0.5, 0.75):
            ax.plot(x, y + 0.08 * np.sin(x * 2 * np.pi), color=color, lw=lw)
    elif kind == "wind":        # gust lines with curled ends
        for y, L in ((0.72, 0.62), (0.5, 0.82), (0.28, 0.5)):
            ax.plot([0.05, L], [y, y], color=color, lw=lw)
            th = np.linspace(-np.pi / 2, np.pi, 30)
            ax.plot(L + 0.09 * np.cos(th), y + 0.09 + 0.09 * np.sin(th),
                    color=color, lw=lw)
    elif kind == "sun":
        ax.add_patch(plt.Circle((0.5, 0.5), 0.2, fill=False, color=color, lw=lw))
        for a in np.arange(8) * np.pi / 4:
            ax.plot([0.5 + 0.29 * np.cos(a), 0.5 + 0.42 * np.cos(a)],
                    [0.5 + 0.29 * np.sin(a), 0.5 + 0.42 * np.sin(a)], color=color, lw=lw)
    elif kind == "cloud":
        for cx, cy, r in ((0.35, 0.45, 0.18), (0.58, 0.55, 0.24), (0.75, 0.42, 0.15)):
            ax.add_patch(plt.Circle((cx, cy), r, fill=False, color=color, lw=lw))
    elif kind == "calendar":
        ax.add_patch(Rectangle((0.12, 0.10), 0.76, 0.72, fill=False, color=color, lw=lw))
        ax.add_patch(Rectangle((0.12, 0.64), 0.76, 0.18, color=color))
        for xx in (0.32, 0.68):
            ax.plot([xx, xx], [0.74, 0.94], color=color, lw=lw + 1)
        for gx in (0.27, 0.5, 0.73):
            for gy in (0.25, 0.45):
                ax.add_patch(Rectangle((gx - 0.05, gy - 0.05), 0.1, 0.1, color=color))


def pipe_match():
    _, row = _pipe_inputs()
    wx = row.weather_simple.lower().replace("cloudy_overcast", "cloudy")
    items = [
        ("tide", "Tide", f"{row.tide_ft:.1f} ft"),
        ("surf", "Surf", f"{row.surf_min_ft:g}-{row.surf_max_ft:g} ft"),
        ("swell", "Swell", f"{row.primary_swell_height_ft:.1f} ft at {row.primary_swell_period_s:g} s"),
        ("wind", "Wind", f"{row.wind_speed_mph:.1f} mph" if row.wind_speed_mph < 1
         else f"{row.wind_speed_mph:.0f} mph"),
        ("sun" if wx == "clear" else "cloud", "Weather",
         f"{row.real_temperature_f:.0f} °F, {wx}"),
        ("calendar", "Day", f"{pd.Timestamp(PIPE_DATE):%A}"
         + (", weekend" if row.is_weekend else ", weekday")),
    ]
    fig = plt.figure(figsize=(11, 3.4), facecolor=BG)
    for i, (icon, name, val) in enumerate(items):
        c, r = i % 2, i // 2
        x0, y0 = 0.03 + c * 0.50, 0.70 - r * 0.32
        _icon(fig.add_axes([x0, y0, 0.06, 0.22]), icon)
        fig.text(x0 + 0.09, y0 + 0.11, name, color=INK_DIM, fontsize=13, va="center")
        fig.text(x0 + 0.20, y0 + 0.11, val, color=INK, fontsize=14, va="center",
                 fontweight="bold")
    _save(fig, "pipeline_4_match.png")


def pipe_forecast():
    flog = pd.read_csv(FLOG)
    day = flog[(flog.date == PIPE_DATE) & (flog.forecast_kind == "day_ahead")].copy()
    day["h"] = day.hour_local.str[:2].astype(int)
    day = day.sort_values("h")
    fig, ax = plt.subplots(figsize=(11, 4.6), facecolor=BG)
    _style(ax)
    ax.fill_between(day.h, day.lower_q10, day.upper_q90, color=AQUA, alpha=0.22,
                    linewidth=0)
    ax.plot(day.h, day.predicted, color=AQUA, linewidth=2.2)
    sc = day[day.actual.notna()]
    ax.scatter(sc.h, sc.actual, s=40, color=LIME, edgecolors=BG, linewidths=1.0, zorder=4)
    hh = int(PIPE_TIME[:2])
    ax.axvline(hh, color=INK_DIM, linewidth=1.0, linestyle=(0, (3, 3)))
    ax.text(hh + 0.15, 0.97, f"the {hh % 12 or 12}{'am' if hh < 12 else 'pm'} hour",
            color=INK_DIM, fontsize=9.5,
            transform=ax.get_xaxis_transform(), va="top")
    hrs = list(range(day.h.min(), day.h.max() + 1, 2))
    ax.set_xticks(hrs)
    ax.set_xticklabels([f"{h % 12 or 12}{'am' if h < 12 else 'pm'}" for h in hrs])
    ax.set_xlim(day.h.min() - 0.3, day.h.max() + 0.3)
    top = max(day.upper_q90.max(), sc.actual.max()) * 1.12
    ax.set_ylim(-0.03 * top, top)      # room for a dot sitting on zero
    ax.set_ylabel("surfers", color=INK, fontsize=10.5)
    # Direct labels, top-left, in place of a legend box
    for k, (txt, col) in enumerate((("forecast, made the evening before", AQUA),
                                    ("likely range", INK_DIM),
                                    ("counted on the day", LIME))):
        ax.text(0.015, 0.95 - k * 0.085, txt, color=col, fontsize=9.5,
                transform=ax.transAxes, va="top")
    fig.tight_layout()
    _save(fig, "pipeline_5_forecast.png")
    r = day[day.h == hh].iloc[0]
    return dict(pred=round(r.predicted, 1), lo=round(r.lower_q10, 1),
                hi=round(r.upper_q90, 1), actual=r.actual,
                day_mae=round((sc.predicted - sc.actual).abs().mean(), 2))


def pipeline_steps():
    return dict(watch=pipe_watch(), spot=pipe_spot(), count=pipe_count(),
                match=pipe_match(), forecast=pipe_forecast())


# --------------------------------------------------------------------------
def tide_bars(step=1.0, min_hours=20):
    """Median count per 1-ft tide bin: the tide relationship without the
    hour-to-hour scatter that buries it in a point cloud."""
    t = _daylight()
    lo = np.floor(t.tide_ft.min()); hi = np.ceil(t.tide_ft.max())
    edges = np.arange(lo, hi + step, step)
    t["bin"] = pd.cut(t.tide_ft, edges, right=False)
    g = t.groupby("bin", observed=True).surfer_count.agg(["median", "size"])
    g = g[g["size"] >= min_hours]
    labels = [f"{iv.left:g} to {iv.right:g}" for iv in g.index]
    fig, ax = plt.subplots(figsize=(9.6, 4.8), facecolor=BG)
    _style(ax); ax.grid(axis="x", visible=False)
    x = np.arange(len(g))
    ax.bar(x, g["median"], width=0.72, color=AQUA, zorder=3)
    top = g["median"].max() * 1.2
    for xi, (m, n) in zip(x, g.values):
        ax.text(xi, m + top * 0.015, f"{m:.0f}", ha="center", va="bottom",
                color=INK, fontsize=10, fontweight="bold")
        ax.text(xi, -0.11, f"{int(n):,} hours", transform=ax.get_xaxis_transform(),
                ha="center", va="top", color=INK_DIM, fontsize=8.5)
    ax.set_ylim(0, top)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.tick_params(axis="x", colors=INK, labelsize=9.5)
    ax.set_xlabel("tide height, ft", color=INK, fontsize=10.5, labelpad=20)
    ax.set_ylabel("typical surfers in the water", color=INK, fontsize=10.5)
    ax.set_title("Low tide brings the crowd", color=INK, fontsize=13.5,
                 fontweight="bold", loc="left", pad=26)
    ax.text(0, 1.035, f"Median count per daylight hour, by tide height  ·  "
            f"{len(t):,} hours, {t.date.min()} to {t.date.max()}",
            transform=ax.transAxes, color=INK_DIM, fontsize=9)
    fig.tight_layout()
    _save(fig, "tide_vs_count.png")
    return {l: (float(m), int(n)) for l, (m, n) in zip(labels, g.values)}


# --------------------------------------------------------------------------
def weekend_by_month(min_days=2, tide_max=3.0):
    """Weekend vs weekday, month by month, as the mean of each day's mean count,
    with the weekend/weekday ratio labelled so the sentence in the README can
    be checked against it. Daylight hours with the tide below tide_max only
    (as of 2026-10-23), like when_to_go: tide moves the crowd more than the
    calendar does, and a month whose weekends happened to fall on high tides
    would otherwise read as a weak weekend effect."""
    t = _daylight()
    t = t[t.tide_ft < tide_max]
    daily = t.groupby("date").agg(m=("surfer_count", "mean"),
                                  we=("is_weekend", "first")).reset_index()
    daily["ym"] = daily.date.str[:7]
    g = daily.groupby(["ym", "we"]).m.agg(["mean", "size"]).unstack("we")
    months = list(g.index)
    fig, ax = plt.subplots(figsize=(11, 4.8), facecolor=BG)
    _style(ax); ax.grid(axis="x", visible=False)
    x = np.arange(len(months)); w = 0.38
    ratios = {}
    for k, (flag, col, name) in enumerate(((False, AQUA, "weekday"), (True, LIME, "weekend"))):
        v = g[("mean", flag)].where(g[("size", flag)] >= min_days)
        ax.bar(x + (k - 0.5) * w, v, width=w - 0.04, color=col, zorder=3, label=name)
    for xi, m in zip(x, months):
        a, b = g.loc[m, ("mean", False)], g.loc[m, ("mean", True)]
        if g.loc[m, ("size", False)] >= min_days and g.loc[m, ("size", True)] >= min_days:
            ratios[m] = b / a
            ax.text(xi, max(a, b) + 0.6, f"{b / a:.1f}x", ha="center", va="bottom",
                    color=INK_DIM, fontsize=8.5)
    ax.set_xticks(x)
    # Year shown wherever it changes: the record skips months, so "January"
    # is not a reliable place to mark a new year.
    ax.set_xticklabels([pd.Period(m).strftime("%b\n%Y") if (i == 0 or m[:4] != months[i - 1][:4])
                        else pd.Period(m).strftime("%b") for i, m in enumerate(months)])
    ax.set_ylabel("average surfers per hour", color=INK, fontsize=10.5)
    ax.set_ylim(0, np.nanmax(g["mean"].values) * 1.18)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=9.5, ncol=2)
    for txt in leg.get_texts():
        txt.set_color(INK)
    ax.set_title("Weekends against weekdays, month by month", color=INK, fontsize=13.5,
                 fontweight="bold", loc="left", pad=46)
    ax.text(0, 1.035, f"Hours with the tide below {tide_max:g} ft only: {len(t):,} hours over "
            f"{daily.date.nunique()} days, {daily.date.min()} to {daily.date.max()}.\n"
            f"Each bar averages each day's mean count; the ratio above each pair is "
            f"weekend / weekday.", transform=ax.transAxes, color=INK_DIM, fontsize=10,
            linespacing=1.4)
    fig.tight_layout()
    _save(fig, "weekend_by_month.png")
    return {m: round(r, 2) for m, r in ratios.items()}


# --------------------------------------------------------------------------
def surfer_poses():
    j = json.load(open(CVAT / POSE_SET / "annotations" / "instances_default.json"))
    iid = next(i["id"] for i in j["images"] if i["file_name"].endswith(POSE_FRAME))
    best = {}
    for a in j["annotations"]:
        p = a["attributes"].get("pose")
        if a["image_id"] == iid and p in ("standing", "sitting", "prone"):
            if p == "standing" and round(a["bbox"][0]) != POSE_STANDING_X:
                continue
            area = a["bbox"][2] * a["bbox"][3]
            if area > best.get(p, (0, None))[0]:
                best[p] = (area, a["bbox"])
    img = _rgb(CROPS / POSE_FRAME)

    # Laid out in inches rather than with a gridspec, so the strip spans exactly
    # the width of the three panels and their gaps: an equal-aspect image in a
    # gridspec row shrinks to the row's height and leaves margins either side.
    FW, MARGIN, GAP = 11.0, 0.33, 0.35
    W = FW - 2 * MARGIN
    strip_h = W * img.shape[0] / img.shape[1]
    pw = (W - 2 * GAP) / 3
    ph = pw * 240 / 384
    head, mid, foot = 0.95, 0.85, 0.15
    FH = head + strip_h + mid + ph + foot
    fig = plt.figure(figsize=(FW, FH), facecolor=BG)
    top = fig.add_axes([MARGIN / FW, (foot + ph + mid) / FH, W / FW, strip_h / FH])
    top.imshow(img, aspect="auto"); _bare(top)
    fig.text(MARGIN / FW, 1 - 0.40 / FH, "Why some surfers are hard to count", color=INK,
             fontsize=13.5, fontweight="bold")
    fig.text(MARGIN / FW, 1 - 0.68 / FH, "Three hand-labeled surfers from one frame, enlarged 6x.  "
             "Riders stand out; a surfer lying on the board is a smudge a few "
             "pixels high.", color=INK_DIM, fontsize=9.5)

    order = [("standing", "Standing", "riding a wave"),
             ("sitting", "Sitting", "waiting for a wave"),
             ("prone", "Lying down", "paddling")]
    for k, (pose, name, what) in enumerate(order):
        x, y, w, h = best[pose][1]
        top.add_patch(Rectangle((x - 3, y - 3), w + 6, h + 6, fill=False,
                                edgecolor=LIME, linewidth=1.4))
        # sitting and lying sit close together here: one label above its
        # box, the other below, so they never overlap
        below = pose == "sitting"
        top.text(x + w / 2, y + h + 6 if below else y - 6, name.split()[0].lower(),
                 color=LIME, fontsize=8, ha="center", va="top" if below else "bottom")
        # a fixed 64x40 window centred on the box, so every panel is to scale
        cx, cy = x + w / 2, y + h / 2
        x0 = int(np.clip(cx - 32, 0, img.shape[1] - 64))
        y0 = int(np.clip(cy - 20, 0, img.shape[0] - 40))
        patch = cv2.resize(img[y0:y0 + 40, x0:x0 + 64], (384, 240),
                           interpolation=cv2.INTER_LANCZOS4)
        ax = fig.add_axes([(MARGIN + k * (pw + GAP)) / FW, foot / FH, pw / FW, ph / FH])
        ax.imshow(patch, aspect="auto"); _bare(ax)
        ax.set_title(f"{name}", color=INK, fontsize=12, fontweight="bold",
                     loc="left", pad=16)
        ax.text(0, 1.025, f"{what}  ·  {w:.0f}x{h:.0f} px box", color=INK_DIM,
                fontsize=9, transform=ax.transAxes)
    _save(fig, "surfer_poses.png")


# --------------------------------------------------------------------------
def when_to_go(min_hours=5, tide_max=3.0):
    t = _daylight()
    # Tide moves the crowd more than anything else, and it cycles through the
    # week independently of the calendar; holding it to the busy range keeps
    # a low-tide Tuesday from reading as a busy-Tuesday pattern.
    t = t[t.tide_ft < tide_max]
    t["h"] = t.hour_local.astype(int)
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    hours = list(range(6, 20))
    t = t[t.h.isin(hours)]
    med = t.pivot_table(index="day_of_week", columns="h", values="surfer_count",
                        aggfunc="median").reindex(index=days, columns=hours)
    cnt = t.pivot_table(index="day_of_week", columns="h", values="surfer_count",
                        aggfunc="size").reindex(index=days, columns=hours).fillna(0)
    shown = med.where(cnt >= min_hours)

    cmap = LinearSegmentedColormap.from_list("aqua", ["#0d2a30", AQUA, "#d9f4f8"])
    cmap.set_bad(NODATA)
    fig, ax = plt.subplots(figsize=(11, 4.9), facecolor=BG)
    ax.set_facecolor(BG)
    vmax = np.nanpercentile(shown.values, 98)
    im = ax.imshow(np.ma.masked_invalid(shown.values), cmap=cmap, vmin=0, vmax=vmax,
                   aspect="auto")
    for i in range(len(days)):
        for k in range(len(hours)):
            v = shown.values[i, k]
            if np.isnan(v):
                continue
            ax.text(k, i, f"{v:.0f}", ha="center", va="center", fontsize=9,
                    color="black" if v > vmax * 0.55 else INK)
    # 2px surface gap between cells
    ax.set_xticks(np.arange(-.5, len(hours)), minor=True)
    ax.set_yticks(np.arange(-.5, len(days)), minor=True)
    ax.grid(which="minor", color=BG, linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.set_xticks(range(len(hours)))
    ax.set_xticklabels([f"{h % 12 or 12}{'am' if h < 12 else 'pm'}" for h in hours])
    ax.set_yticks(range(len(days))); ax.set_yticklabels(days)
    ax.tick_params(colors=INK_DIM, labelsize=9.5, length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("When is it busy?  Typical surfers in the water, by hour and day",
                 color=INK, fontsize=13.5, fontweight="bold", loc="left", pad=26)
    ax.text(0, 1.035,
            f"Median count of {len(t):,} counted hours over {t.date.nunique()} days, "
            f"{t.date.min()} to {t.date.max()}, tide below {tide_max:g} ft only.  "
            f"Dark gray: fewer than {min_hours} hours recorded.",
            transform=ax.transAxes, color=INK_DIM, fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cb.outline.set_edgecolor(GRID); cb.ax.tick_params(colors=INK_DIM, labelsize=8.5)
    cb.set_label("surfers", color=INK_DIM, fontsize=9)
    fig.tight_layout()
    _save(fig, "when_to_go.png")
    return len(t), t.date.nunique()


# --------------------------------------------------------------------------
def seasons():
    t = _daylight()
    t["ym"] = t.date.str[:7]
    g = t.groupby("ym").agg(n=("surfer_count", "size"), days=("date", "nunique"),
                            med=("surfer_count", "median"),
                            empty=("surfer_count", lambda s: 100 * (s == 0).mean()))
    months = pd.period_range(t.date.min()[:7], t.date.max()[:7], freq="M")
    labels = [m.strftime("%b\n%Y") if (m.month in (1, 10) or i == 0) else m.strftime("%b")
              for i, m in enumerate(months)]
    keys = [str(m) for m in months]
    g = g.reindex(keys)
    x = np.arange(len(keys))

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7.4), facecolor=BG, sharex=True)
    for ax, col, color, ylab in ((a1, "med", AQUA, "typical surfers per hour"),
                                 (a2, "empty", LIME, "hours with nobody out, %")):
        _style(ax)
        ok = g[col].notna().values
        ax.bar(x[ok], g[col].values[ok], width=0.72, color=color, zorder=3)
        for xi in x[~ok]:
            ax.axvspan(xi - 0.4, xi + 0.4, color=NODATA, zorder=1)
        ax.set_ylabel(ylab, color=INK, fontsize=10)
        top = np.nanmax(g[col].values) * 1.25
        ax.set_ylim(0, top)
        for xi, v in zip(x[ok], g[col].values[ok]):
            ax.text(xi, v + top * 0.02, f"{v:.0f}", ha="center", va="bottom",
                    color=INK_DIM, fontsize=9)
    for xi in x[~g.n.notna().values]:
        a1.text(xi, a1.get_ylim()[1] * 0.5, "no\ndata", ha="center", va="center",
                color=INK_DIM, fontsize=8.5)
    for xi, d in zip(x, g.days.values):
        if not np.isnan(d):
            a2.text(xi, -0.20, f"{d:.0f} days", transform=a2.get_xaxis_transform(),
                    ha="center", va="top", color=INK_DIM, fontsize=8)
    a2.set_xticks(x); a2.set_xticklabels(labels)
    a2.tick_params(axis="x", pad=2)
    # Month labels on the top panel too, so it reads without looking down past
    # the bottom one (sharex hides them by default).
    a1.tick_params(axis="x", labelbottom=True, pad=2)
    a1.set_title("Through the year: crowds thin and empty hours climb into winter",
                 color=INK, fontsize=13.5, fontweight="bold", loc="left", pad=38)
    a1.text(0, 1.04, "Median count per daylight hour, and the share of hours with "
            "nobody out, by month.  Gaps are months the camera was not recorded.\n"
            "Counts before 2026-09-21 come from the earlier detector, which missed more "
            "surfers in fog, so the winter-2025 empty share may run high.",
            transform=a1.transAxes, color=INK_DIM, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save(fig, "seasons.png")
    return g


# --------------------------------------------------------------------------
def forecast_within_n(max_n=15):
    f = pd.read_csv(FLOG)
    s = f[f.actual.notna()].copy()
    s["err"] = (s.predicted - s.actual).abs()
    series = [
        ("the night before", s[s.forecast_kind == "day_ahead"], AQUA),
        ("a week ahead", s[(s.forecast_kind == "week") & (s.lead_days == 7)], LIME),
    ]
    ns = np.arange(0, max_n + 1)
    fig, ax = plt.subplots(figsize=(9.6, 5.4), facecolor=BG)
    _style(ax)
    out = {}
    for (name, d, color), dy in zip(series, (3, -7)):
        share = [100 * (d.err <= n).mean() for n in ns]
        out[name] = (len(d), share)
        ax.plot(ns, share, color=color, linewidth=2.2, marker="o", markersize=5,
                markeredgecolor=BG, zorder=3)
        ax.text(ns[-1] + 0.3, share[-1] + dy, f"{name}\n({len(d):,} hours)", color=color,
                fontsize=9.5, va="center")
    day = out["the night before"][1]
    for n in (3, 5, 10, 15):
        ax.annotate(f"{day[n]:.0f}% within {n}", (n, day[n]), textcoords="offset points",
                    xytext=(-8, 12), ha="right", color=INK_DIM, fontsize=9)
    ax.set_xlim(0, max_n + 3.2); ax.set_ylim(0, 106)
    ax.set_yticks(range(0, 101, 20))
    ax.set_xticks(range(0, max_n + 1, 1))
    ax.set_xlabel("forecast missed by no more than this many surfers", color=INK,
                  fontsize=10.5)
    ax.set_ylabel("share of forecast hours, %", color=INK, fontsize=10.5)
    ax.set_title("How close does the forecast get?", color=INK, fontsize=13.5,
                 fontweight="bold", loc="left", pad=26)
    ax.text(0, 1.035, f"Every forecast is logged before the day and scored against the "
            f"detector's count afterwards  ·  {s.date.min()} to {s.date.max()}",
            transform=ax.transAxes, color=INK_DIM, fontsize=9)
    fig.tight_layout()
    _save(fig, "forecast_within_n.png")
    return out


# --------------------------------------------------------------------------
def crowd_vs_rating(tide_lo=0.0, tide_hi=3.0, min_days=4):
    t = _daylight()
    t = t[(t.tide_ft >= tide_lo) & (t.tide_ft <= tide_hi)]
    names = {1: "Poor", 2: "Poor to fair", 3: "Fair", 4: "Fair to good"}
    g = t.groupby("rating_value")
    keep = [r for r, d in g if d.date.nunique() >= min_days]
    dropped = {names.get(r, r): (len(d), d.date.nunique()) for r, d in g if r not in keep}

    fig, ax = plt.subplots(figsize=(8.6, 5.2), facecolor=BG)
    _style(ax)
    data = [g.get_group(r).surfer_count.values for r in keep]
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color=LIME, linewidth=2.2),
                    whiskerprops=dict(color=INK_DIM), capprops=dict(color=INK_DIM),
                    boxprops=dict(facecolor=AQUA + "55", edgecolor=AQUA))
    for i, (r, d) in enumerate(zip(keep, data), start=1):
        ax.text(i, np.median(d), f" {np.median(d):.0f}", color=LIME, fontsize=10,
                va="bottom", ha="left", fontweight="bold")
        ax.text(i, -0.09, f"{len(d)} hours, {g.get_group(r).date.nunique()} days",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                color=INK_DIM, fontsize=8.5)
    ax.set_xticks(range(1, len(keep) + 1))
    ax.set_xticklabels([names.get(r, r) for r in keep])
    ax.tick_params(axis="x", colors=INK, labelsize=10.5, pad=4)
    ax.set_ylabel("surfers counted that hour", color=INK, fontsize=10.5)
    ax.set_title("Does better surf bring a bigger crowd?", color=INK, fontsize=13.5,
                 fontweight="bold", loc="left", pad=40)
    note = (f"Surfline's rating for the hour, tide {tide_lo:g}-{tide_hi:g} ft only "
            "(tide matters more than anything else).")
    if dropped:
        note += "\nNot shown, too few days: " + ", ".join(
            f"{k} ({h} hours, {d} days)" for k, (h, d) in dropped.items()) + "."
    ax.text(0, 1.035, note, transform=ax.transAxes, color=INK_DIM, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, "crowd_vs_rating.png")
    return {names.get(r, r): (len(d), float(np.median(d))) for r, d in zip(keep, data)}, dropped


FIGS = {"pipeline": pipeline_steps, "poses": surfer_poses, "when": when_to_go,
        "seasons": seasons, "within": forecast_within_n, "rating": crowd_vs_rating,
        "tide": tide_bars, "weekend": weekend_by_month}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="*", choices=sorted(FIGS))
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for k in (args.only or FIGS):
        r = FIGS[k]()
        if r is not None:
            print(f"  {k}: {r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
