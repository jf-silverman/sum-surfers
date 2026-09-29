"""
Evening email: the day's forecast with the day's actual counts drawn over it.

Answers one question each night — how did today's forecast actually do? The
chart shows the forecast (median line and 80% band) with the real hourly
detection counts plotted on top, so the misses are visible rather than
summarized.

Where the forecast comes from matters, and the script is explicit about it:

- **Recorded** (`data/forecasts/forecast_<date>.csv`): written by
  `plot_daily_prediction.py` when the forecast was made, before the day
  happened. This is the real thing and is used whenever it exists.
- **Reconstructed** (`--allow-reconstruct`): no record exists, so a model is
  fit on rows strictly *before* the target date and asked to predict it. That
  is a re-enactment, not the original forecast — a model fit today has
  seen months the original had not. Charts and emails built this way say so.

Only days that already have detections are reported on, so an evening run after
the pipeline has counted the day works without special-casing.

Usage:
    python code/email_daily_report.py                  # most recent counted day
    python code/email_daily_report.py --date 2026-09-21
    python code/email_daily_report.py --no-send        # build the chart only
"""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_daily_prediction as pdp  # noqa: E402
from send_email import send_email  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_CSV = _PROJECT_ROOT / "data" / "predictions" / "predictions.csv"
OUT_DIR = _PROJECT_ROOT / "data" / "charts"


def actual_counts(target_date):
    """Hourly detection counts for a date: [(datetime, count), ...]."""
    df = pd.read_csv(PREDICTIONS_CSV, float_precision="round_trip")
    df = df[(df["date"] == target_date.isoformat())
            & (df["quality_ok"].astype(str) == "True")
            & df["surfer_count"].notna()]
    out = []
    for _, r in df.iterrows():
        hh, mm = map(int, str(r["time_local"]).split(":")[:2])
        out.append((datetime(target_date.year, target_date.month, target_date.day, hh, mm),
                    float(r["surfer_count"])))
    return sorted(out)


def most_recent_counted_day(lookback=30):
    df = pd.read_csv(PREDICTIONS_CSV, float_precision="round_trip")
    df = df[(df["quality_ok"].astype(str) == "True") & df["surfer_count"].notna()]
    if df.empty:
        return None
    today = datetime.now().date()
    dates = sorted({datetime.strptime(d, "%Y-%m-%d").date() for d in df["date"]}, reverse=True)
    for d in dates:
        if (today - d).days <= lookback:
            return d
    return dates[0]


def load_recorded_forecast(target_date):
    path = pdp.FORECASTS_DIR / f"forecast_{target_date.isoformat()}.csv"
    if not path.exists():
        return None
    fc = pd.read_csv(path)
    rows = []
    for _, r in fc.iterrows():
        hh, mm = map(int, str(r["hour_local"]).split(":")[:2])
        rows.append(dict(
            hour=datetime(target_date.year, target_date.month, target_date.day, hh, mm),
            point=float(r["predicted"]), lower=float(r["lower_q10"]), upper=float(r["upper_q90"])))
    made_at = str(fc["forecast_made_at"].iloc[0]) if "forecast_made_at" in fc else "unknown"
    return rows, made_at


def reconstruct_forecast(target_date):
    """Refit on rows strictly before target_date, then predict its hours.

    Deliberately excludes the target day so the reconstruction cannot score
    itself on data it trained on. It still is not the original forecast: this
    model has seen everything up to yesterday, where the original saw only up to
    its own run date. Used only when no recorded forecast exists.
    """
    import fit_surfer_count_model as fit  # noqa: PLC0415

    X, y, df, _numeric_cols = fit.load_and_prepare()
    mask = pd.to_datetime(df["date"]).dt.date < target_date
    if mask.sum() < 100:
        raise RuntimeError(f"Only {int(mask.sum())} rows before {target_date} — too few to refit.")

    X_train, y_train = X[mask], np.asarray(y[mask], dtype=float)
    models = {lv: fit.fit_quantile_model_robust(X_train, y_train, lv, allow_degenerate=True)[0]
              for lv in pdp.FAN_LEVELS}

    # Feature rows for the target day come from the same predictor map the chart
    # uses, so a reconstruction and a real forecast are built the same way.
    rows = []
    day_rows = df[pd.to_datetime(df["date"]).dt.date == target_date]
    if day_rows.empty:
        raise RuntimeError(f"No feature rows for {target_date}; cannot reconstruct.")
    Xd = X[pd.to_datetime(df["date"]).dt.date == target_date]
    for (_, frow), (_, xrow) in zip(day_rows.iterrows(), Xd.iterrows()):
        hour = int(frow["hour_local"])
        x1 = xrow.to_frame().T
        q = {}
        running = 0.0
        for lv in pdp.FAN_LEVELS:
            running = max(running, float(models[lv].predict(x1)[0]))
            q[lv] = running
        rows.append(dict(hour=datetime(target_date.year, target_date.month, target_date.day, hour, 0),
                         point=q[0.50], lower=q[0.10], upper=q[0.90]))
    return sorted(rows, key=lambda r: r["hour"]), None


FEATURES_CSV = _PROJECT_ROOT / "data" / "training_features.csv"


def tide_by_hour(target_date):
    """[(datetime, tide_ft), ...] for the day, from the training features table.

    Tide is already joined to every counted hour there, so it needs no second
    API call and is exactly the value the forecast model saw.
    """
    if not FEATURES_CSV.exists():
        return []
    df = pd.read_csv(FEATURES_CSV, float_precision="round_trip")
    df = df[(df["date"] == target_date.isoformat()) & df["tide_ft"].notna()]
    if df.empty:
        return []
    # One row per hour: duplicate hours (two clips in the same hour) carry the
    # same tide, so taking the first is enough.
    seen, out = set(), []
    for _, r in df.sort_values("hour_local").iterrows():
        hour = int(r["hour_local"])
        if hour in seen:
            continue
        seen.add(hour)
        out.append((datetime(target_date.year, target_date.month, target_date.day, hour, 0),
                    float(r["tide_ft"])))
    return out


# The crops the detector runs on are a 1280x180 strip cut from y=420 of the
# camera's 1280x720 frame (ROI_* in get_cropped_frame.py). That strip is all the
# model ever sees, and until 2026-09-28 it was also all the email showed, which
# made the attached frames hard to place -- a band of water with no beach, no
# horizon and no sense of where along the point you were looking.
#
# The full frame is not saved anywhere, but the clip it was cut from is, so it
# can be re-extracted. Boxes are produced in strip coordinates and shifted down
# by ROI_Y to land correctly on the full frame.
CLIPS_DIR = _PROJECT_ROOT / "data" / "not_needed_in_repo" / "surf_clips"
ROI_X, ROI_Y, ROI_W, ROI_H = 0, 420, 1280, 180
CLIP_FRAME_TIME_SEC = 2.5          # matches FRAME_TIME_SEC in get_cropped_frame.py


def full_frame_for(target_date, time_local):
    """The uncropped camera frame behind a crop, re-read from its clip.

    Returns None when the clip is gone -- manage_clips.py deletes old ones once
    storage passes its limit -- so callers fall back to the strip rather than
    losing the attachment entirely.
    """
    hh, mm = str(time_local).split(":")[:2]
    clip = CLIPS_DIR / target_date.isoformat() / f"{hh}_{mm}" / "clip.mp4"
    if not clip.exists():
        return None
    cap = cv2.VideoCapture(str(clip))
    cap.set(cv2.CAP_PROP_POS_MSEC, CLIP_FRAME_TIME_SEC * 1000)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def draw_on_full_frame(frame, boxes):
    """Detector boxes on the full frame, with the ROI strip outlined.

    The outline matters: without it a reader sees surfers up the beach that the
    model did not box and concludes it missed them, when in fact they are simply
    outside the band it is given. Drawing the boundary makes 'not looked at'
    visibly different from 'looked at and missed'.
    """
    out = frame.copy()
    # Dashed rather than solid: a continuous line reads as part of the scene
    # (the horizon, a wire), while a dashed one reads as an annotation.
    y0, y1 = ROI_Y, ROI_Y + ROI_H
    for x in range(ROI_X, ROI_X + ROI_W, 24):
        for yy in (y0, y1):
            cv2.line(out, (x, yy), (min(x + 12, ROI_X + ROI_W), yy), (250, 250, 250), 1,
                     cv2.LINE_AA)
    label = "detector's view"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(out, (ROI_X + 8, y0 - th - 10), (ROI_X + 14 + tw, y0 - 2),
                  (0, 0, 0), -1)
    cv2.putText(out, label, (ROI_X + 11, y0 - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250, 250, 250), 1, cv2.LINE_AA)
    overlay = out.copy()
    visible = 0
    for bx1, by1, bx2, by2, _conf in boxes:
        visible += 1
        cv2.rectangle(overlay, (int(bx1), int(by1) + ROI_Y),
                      (int(bx2), int(by2) + ROI_Y), (90, 227, 157), 2)
    out = cv2.addWeighted(overlay, 0.85, out, 0.15, 0)
    return out, visible


def extreme_frames(target_date, model=None, forecast_rows=None):
    """Detector frames at the day's telling hours.

    Three are about where COUNTING is hardest to trust — the busiest hour, an
    empty hour, and the quietest non-empty one. The fourth, added 2026-09-28, is
    about where FORECASTING went wrong: the hour whose predicted count missed the
    counted one by the most. That is the frame worth actually looking at, because
    it is the one place a bad number and the picture behind it can be compared
    directly, which separates "the model was wrong" from "the detector was wrong".

    `forecast_rows` is optional; without it the divergence frame is simply
    omitted rather than failing, since some days have no recorded forecast.

    Returns [(label, path, visible_count, stamp), ...].
    """
    df = pd.read_csv(PREDICTIONS_CSV, float_precision="round_trip")
    df = df[(df["date"] == target_date.isoformat())
            & (df["quality_ok"].astype(str) == "True")
            & df["surfer_count"].notna()]
    if df.empty:
        return []

    picks = []          # (label, row, extra banner text)
    busiest = df.loc[df["surfer_count"].idxmax()]
    picks.append(("Busiest hour", busiest, ""))
    zeros = df[df["surfer_count"] == 0]
    if not zeros.empty:
        picks.append(("Empty hour", zeros.iloc[0], ""))
    nonzero = df[df["surfer_count"] > 0]
    if not nonzero.empty:
        quietest = nonzero.loc[nonzero["surfer_count"].idxmin()]
        if quietest["filename"] != busiest["filename"]:
            picks.append(("Quietest hour above zero", quietest, ""))

    # The hour the forecast missed by the most. Matched on the hour number, the
    # same way the hour-by-hour table matches, so a clip taken at 10:02 is
    # scored against the 10:00 forecast.
    if forecast_rows:
        by_hour = {r["hour"].hour: r for r in forecast_rows}
        best_gap, best_row, best_f = -1.0, None, None
        for _i, row in df.iterrows():
            hr = int(str(row["time_local"]).split(":")[0])
            f = by_hour.get(hr)
            if f is None:
                continue
            gap = abs(f["point"] - row["surfer_count"])
            if gap > best_gap:
                best_gap, best_row, best_f = gap, row, f
        already = {r["filename"] for _l, r, _e in picks}
        if best_row is not None and best_row["filename"] not in already and best_gap > 0:
            direction = "over" if best_f["point"] > best_row["surfer_count"] else "under"
            picks.append((f"Biggest forecast miss", best_row,
                          f"  |  forecast {best_f['point']:.1f}, {direction} by {best_gap:.1f}"))

    if model is None:
        model = pdp.ds.load_model()

    out = []
    for label, row, extra in picks:
        img_path = pdp.ds.CROPS_DIR / row["filename"]
        if not img_path.exists():
            continue
        # Prefer the whole camera frame; fall back to the strip if the clip has
        # been cleaned up.
        boxes = pdp.ds.run_inference_with_boxes(model, img_path)
        full = full_frame_for(target_date, row["time_local"])
        if full is not None:
            frame, visible = draw_on_full_frame(full, boxes)
        else:
            frame, visible = pdp.render_detection_frame(img_path, model, boxes=boxes)
        if frame is None:
            continue
        banner_h = 44
        h, w = frame.shape[:2]
        canvas = np.zeros((h + banner_h, w, 3), dtype=np.uint8)
        canvas[:h] = frame
        hh, mm = map(int, str(row["time_local"]).split(":")[:2])
        stamp = datetime(target_date.year, target_date.month, target_date.day,
                         hh, mm).strftime("%-I:%M %p")
        cv2.putText(canvas, f"{label}  |  {stamp}  |  {visible} in this frame "
                    f"(hour counted {row['surfer_count']:.0f}){extra}",
                    (10, h + 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (235, 235, 235), 2,
                    cv2.LINE_AA)
        slug = label.lower().split()[0]
        out_path = OUT_DIR / f"frame_{target_date.isoformat()}_{slug}.png"
        cv2.imwrite(str(out_path), canvas)
        out.append((label, out_path, visible, stamp))
    return out


def missing_predictor_note(target_date):
    """Names the predictor columns that are empty for a date, if any.

    Turns "cannot reconstruct" into something actionable: the reason is almost
    always that an upstream fetch failed for that day, and the columns say which.
    """
    if not FEATURES_CSV.exists():
        return ""
    df = pd.read_csv(FEATURES_CSV, float_precision="round_trip")
    day = df[df["date"] == target_date.isoformat()]
    if day.empty:
        return f"There are no rows at all in the training table for {target_date}."
    empty = [c for c in day.columns if day[c].isna().all()]
    if not empty:
        return ""
    return (f"Every value is missing for these predictors on {target_date}: "
            f"{', '.join(empty)}. Rows missing any predictor are dropped before "
            f"fitting, so this day currently cannot be forecast or trained on. "
            f"The usual cause is a failed forecast fetch during that day's run - "
            f"check the pipeline log for 403s.")


def build_chart(target_date, forecast_rows, actuals, recorded_at, out_path, tide=None):
    fig, ax = plt.subplots(figsize=(13, 6.2), facecolor=pdp.BG_COLOR)
    ax.set_facecolor(pdp.AXES_BG)

    if forecast_rows:
        fh = [r["hour"] for r in forecast_rows]
        ax.fill_between(fh, [r["lower"] for r in forecast_rows],
                        [r["upper"] for r in forecast_rows],
                        color=pdp.AQUA, alpha=0.22, linewidth=0,
                        label="Forecast 80% range")
        ax.plot(fh, [r["point"] for r in forecast_rows], color=pdp.AQUA,
                linewidth=2.5, label="Forecast")

    if actuals:
        ah = [a[0] for a in actuals]
        av = [a[1] for a in actuals]
        # Coral, not lime: lime is the tide line on the right-hand axis, and two
        # lime lines on one chart is exactly the confusion this chart exists to
        # avoid — the whole point is reading crowd against tide.
        ax.plot(ah, av, color=pdp.CORAL, linewidth=2.2, linestyle="-", marker="o",
                markersize=7, markeredgecolor="white", markeredgewidth=0.8,
                label="Actual detected", zorder=5)

    if tide:
        # Second axis on the right, matching the daily chart's convention so the
        # two read the same way. Tide is the strongest single predictor in the
        # model, so a crowd peak sitting on a low tide is the first thing to
        # look for when the forecast misses a busy hour.
        ax2 = ax.twinx()
        ax2.set_facecolor(pdp.AXES_BG)
        ax2.plot([t[0] for t in tide], [t[1] for t in tide], color=pdp.LIME,
                 linestyle="--", linewidth=2, zorder=2, label="Tide (ft)")
        ax2.set_ylabel("Tide (ft)", color=pdp.LIME)
        ax2.tick_params(axis="y", colors=pdp.LIME)
        for spine in ax2.spines.values():
            spine.set_color(pdp.GRID_COLOR)

    ax.set_title(f"Forecast vs. actual — {target_date.strftime('%A, %B %d, %Y')}",
                 color=pdp.TEXT_COLOR)
    ax.set_xlabel("Time", color=pdp.TEXT_COLOR)
    ax.set_ylabel("Surfer count", color=pdp.TEXT_COLOR)
    ax.tick_params(axis="both", colors=pdp.TEXT_COLOR)
    ax.grid(alpha=0.25, color=pdp.GRID_COLOR)
    for spine in ax.spines.values():
        spine.set_color(pdp.GRID_COLOR)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%-I %p"))

    handles, labels = ax.get_legend_handles_labels()
    if tide:
        h2, l2 = ax2.get_legend_handles_labels()
        handles += h2
        labels += l2
    legend = ax.legend(handles, labels, loc="upper left", facecolor=pdp.AXES_BG,
                       edgecolor=pdp.GRID_COLOR)
    for text in legend.get_texts():
        text.set_color(pdp.TEXT_COLOR)

    if not forecast_rows:
        provenance = ("No forecast available for this day — actual counts only "
                      "(see the email body for why)")
    elif recorded_at:
        provenance = f"Forecast recorded {recorded_at}"
    else:
        provenance = ("Forecast RECONSTRUCTED after the fact (no record existed for this "
                      "date) — refit on rows before this day, so not the original forecast")
    fig.text(0.5, 0.01, provenance, fontsize=8, ha="center", va="bottom", color=pdp.MUTED_TEXT)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def summarize(forecast_rows, actuals):
    """Per-hour comparison text, matching each actual to its nearest forecast hour."""
    if not actuals:
        return "No counted hours for this day.", None
    if not forecast_rows:
        lines = [f"  {when.strftime('%-I:%M %p'):>9}   actual {count:>5.0f}"
                 for when, count in actuals]
        return "\n".join(lines), None
    by_hour = {r["hour"].hour: r for r in forecast_rows}
    lines, errs, inside = [], [], 0
    for when, count in actuals:
        f = by_hour.get(when.hour)
        if f is None:
            lines.append(f"  {when.strftime('%-I:%M %p'):>9}   actual {count:>5.0f}   (no forecast for this hour)")
            continue
        err = f["point"] - count
        errs.append(err)
        hit = f["lower"] <= count <= f["upper"]
        inside += int(hit)
        lines.append(f"  {when.strftime('%-I:%M %p'):>9}   actual {count:>5.0f}   "
                     f"forecast {f['point']:>5.1f}   range {f['lower']:.0f}-{f['upper']:.0f}   "
                     f"{'in range' if hit else 'OUTSIDE'}")
    stats = None
    if errs:
        stats = dict(mae=float(np.mean(np.abs(errs))), bias=float(np.mean(errs)),
                     inside=inside, n=len(errs))
    return "\n".join(lines), stats


def hourly_table_html(forecast_rows, actuals):
    """The hour-by-hour comparison as a real HTML table.

    The plain-text version of this is column-aligned with spaces, which only
    survives in a monospace font -- and Gmail renders text/plain proportionally,
    so it arrives as a jumble. This is the same content as a table the client
    will actually lay out. Styles are inline because email clients strip <style>
    blocks, and the palette is the aqua/lime pair used by the project's charts.
    """
    if not actuals:
        return "<p>No counted hours for this day.</p>"

    head = ("<tr>"
            + "".join(f'<th style="padding:6px 10px;text-align:{a};border-bottom:2px solid #444;'
                      f'font-family:system-ui,sans-serif;font-size:13px;">{h}</th>'
                      for h, a in [("Hour", "left"), ("Counted", "right"), ("Forecast", "right"),
                                   ("80% range", "right"), ("", "left")])
            + "</tr>")

    by_hour = {r["hour"].hour: r for r in forecast_rows} if forecast_rows else {}
    rows = []
    for i, (when, count) in enumerate(actuals):
        bg = "#fafafa" if i % 2 else "#ffffff"
        cell = (f'padding:5px 10px;border-bottom:1px solid #e8e8e8;'
                f'font-family:system-ui,sans-serif;font-size:13px;background:{bg};')
        f = by_hour.get(when.hour)
        if f is None:
            rows.append(f'<tr><td style="{cell}">{when.strftime("%-I:%M %p")}</td>'
                        f'<td style="{cell}text-align:right;"><b>{count:.0f}</b></td>'
                        f'<td style="{cell}text-align:right;color:#999;" colspan="3">no forecast for this hour</td></tr>')
            continue
        hit = f["lower"] <= count <= f["upper"]
        # Not colour alone: the word carries the meaning, the colour reinforces it.
        mark = ('<span style="color:#2e7d32;">&#10003; in range</span>' if hit
                else '<span style="color:#c62828;">&#10007; outside</span>')
        rows.append(
            f'<tr><td style="{cell}">{when.strftime("%-I:%M %p")}</td>'
            f'<td style="{cell}text-align:right;"><b>{count:.0f}</b></td>'
            f'<td style="{cell}text-align:right;">{f["point"]:.1f}</td>'
            f'<td style="{cell}text-align:right;color:#666;">{f["lower"]:.0f}&ndash;{f["upper"]:.0f}</td>'
            f'<td style="{cell}font-size:12px;">{mark}</td></tr>')

    return ('<table cellspacing="0" cellpadding="0" '
            'style="border-collapse:collapse;margin:4px 0 2px 0;">'
            + head + "".join(rows) + "</table>")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date", help="YYYY-MM-DD (default: most recent counted day)")
    p.add_argument("--no-send", action="store_true", help="build the chart, do not email")
    p.add_argument("--allow-reconstruct", action="store_true", default=True,
                   help="refit if no recorded forecast exists (default on)")
    p.add_argument("--attach", action="append", default=[],
                   help="extra file to attach; repeatable")
    return p.parse_args()


def main():
    args = parse_args()
    target_date = (datetime.strptime(args.date, "%Y-%m-%d").date() if args.date
                   else most_recent_counted_day())
    if target_date is None:
        print("No counted days found — nothing to report.")
        return 0
    print(f"Reporting on {target_date}")

    actuals = actual_counts(target_date)
    print(f"  {len(actuals)} counted hours")

    recorded = load_recorded_forecast(target_date)
    forecast_note = ""
    if recorded:
        forecast_rows, made_at = recorded
        print(f"  Using the recorded forecast (made {made_at})")
    elif args.allow_reconstruct:
        print("  No recorded forecast for this date — reconstructing from rows before it.")
        try:
            forecast_rows, made_at = reconstruct_forecast(target_date)
        except RuntimeError as e:
            # A missing forecast must not cost the whole report. The counts, the
            # tide and the detector frames are all still worth sending, and the
            # usual cause is an upstream predictor gap that says something in
            # itself — a Surfline endpoint 403 nulls the weather and energy
            # columns for the day, which drops those rows before any model sees
            # them.
            print(f"  Cannot reconstruct a forecast: {e}")
            forecast_rows, made_at = [], None
            forecast_note = (f"No forecast is shown for this day. {e}\n"
                             f"{missing_predictor_note(target_date)}")
    else:
        print("  No recorded forecast and reconstruction disabled — nothing to send.")
        return 1

    tide = tide_by_hour(target_date)
    out_path = OUT_DIR / f"forecast_vs_actual_{target_date.isoformat()}.png"
    build_chart(target_date, forecast_rows, actuals, made_at, out_path, tide=tide)
    print(f"  Chart: {out_path}  (tide points: {len(tide)})")

    # A stable, git-tracked copy for the README's accuracy section, mirroring the
    # latest.png / latest_week.png convention used by plot_daily_prediction.py.
    # The dated file stays the per-day record and remains gitignored; this one is
    # the single published image, overwritten nightly so the README never goes
    # stale. Named so it does not match the forecast_vs_actual_*.png ignore rule.
    latest_path = OUT_DIR / "latest_forecast_vs_actual.png"
    shutil.copyfile(out_path, latest_path)
    print(f"  Published copy: {latest_path}")

    frames = extreme_frames(target_date, forecast_rows=forecast_rows)
    for label, fpath, count, stamp in frames:
        print(f"  {label}: {stamp}, {count} detected -> {fpath.name}")

    table, stats = summarize(forecast_rows, actuals)
    provenance = forecast_note if forecast_note else (f"Forecast recorded {made_at}." if made_at else
                  "NOTE: no forecast was recorded for this date, so the forecast shown was "
                  "reconstructed afterwards by refitting on data from before this day. It is "
                  "a re-enactment, not the original forecast. Days from here on will use the "
                  "real recorded forecast.")
    headline = (f"Average miss {stats['mae']:.1f} surfers, bias {stats['bias']:+.1f} "
                f"({'over' if stats['bias'] > 0 else 'under'}-forecast), "
                f"{stats['inside']} of {stats['n']} hours inside the 80% range."
                if stats else
                (f"{len(actuals)} hours counted, {int(max(a[1] for a in actuals))} at the busiest. "
                 f"No forecast comparison available for this day." if actuals
                 else "No counted hours to report."))

    if frames:
        frame_lines = "\n".join(
            f"  {label}: {stamp}, {count} boxes in the frame shown  ({fpath.name})"
            for label, fpath, count, stamp in frames)
        if not any(label == "Empty hour" for label, _p, _c, _s in frames):
            frame_lines += "\n  (No empty hour today — every counted hour had at least one surfer.)"
    else:
        frame_lines = "  (No frames available for this day.)"

    tide_note = ""
    if tide and actuals:
        peak_hour = max(actuals, key=lambda a: a[1])[0].hour
        tide_at_peak = dict((t[0].hour, t[1]) for t in tide).get(peak_hour)
        lowest = min(tide, key=lambda t: t[1])
        if tide_at_peak is not None:
            tide_note = (f"\nTide: the day's busiest hour ({peak_hour}:00) sat at "
                         f"{tide_at_peak:.2f} ft; the day's low was {lowest[1]:.2f} ft at "
                         f"{lowest[0].hour}:00.\n")

    look = []
    if (OUT_DIR / "latest.png").exists():
        look.append("tomorrow's hourly forecast")
    if (OUT_DIR / "latest_week.png").exists():
        look.append("the week ahead")
    lookahead_note = ("\nAlso attached, looking forward rather than back: "
                      + " and ".join(look) + ".\n") if look else ""

    body = (f"Surf crowd forecast vs. actual for {target_date.strftime('%A, %B %d, %Y')}\n\n"
            f"{headline}\n"
            f"{tide_note}\n"
            f"Hour by hour:\n{table}\n\n"
            f"Detector frames attached, at the three points where counting is hardest\n"
            f"to trust:\n{frame_lines}\n\n"
            f"{provenance}\n\n"
            f"Chart attached, with tide on the right-hand axis. Counts come from the\n"
            f"detector, not a human, so they carry its own error (about 1 surfer per\n"
            f"frame on held-out images).\n"
            f"{lookahead_note}")

    # Tomorrow's hourly forecast and the week grid, both generated by step 9 of
    # the same pipeline run a few minutes before this report. Attached so the
    # evening mail carries what is coming as well as how yesterday went.
    lookahead = [q for q in (OUT_DIR / "latest.png", OUT_DIR / "latest_week.png") if q.exists()]

    html_body = (
        f'<div style="font-family:system-ui,sans-serif;font-size:14px;color:#222;max-width:640px;">'
        f'<h2 style="font-size:17px;margin:0 0 4px 0;">Surf crowd forecast vs. actual</h2>'
        f'<div style="color:#666;font-size:13px;margin-bottom:14px;">'
        f'{target_date.strftime("%A, %B %d, %Y")}</div>'
        f'<p style="margin:0 0 14px 0;"><b>{headline}</b></p>'
        + (f'<p style="margin:0 0 14px 0;color:#444;">{tide_note.strip()}</p>' if tide_note else "")
        + f'<h3 style="font-size:14px;margin:18px 0 6px 0;">Hour by hour</h3>'
        + hourly_table_html(forecast_rows, actuals)
        + f'<h3 style="font-size:14px;margin:20px 0 6px 0;">Attached</h3>'
        + '<ul style="margin:0;padding-left:20px;color:#444;font-size:13px;">'
        + f'<li>Forecast vs. actual chart for {target_date.strftime("%b %d")}, tide on the right axis</li>'
        + ("<li>Tomorrow's hourly forecast (<code>latest.png</code>)</li>"
           if (OUT_DIR / "latest.png").exists() else "")
        + ("<li>The week ahead, hour by day (<code>latest_week.png</code>)</li>"
           if (OUT_DIR / "latest_week.png").exists() else "")
        + "".join(f'<li>{label}: {stamp}, {count} boxes shown</li>'
                  for label, _fp, count, stamp in frames)
        + '</ul>'
        + f'<p style="margin:16px 0 0 0;color:#666;font-size:12px;">{provenance}<br>'
        f'Counts come from the detector, not a human, so they carry its own error '
        f'(about 1 surfer per frame on held-out images).</p></div>')

    # Rebuild the combined forecast-vs-actual log while today's counts are fresh.
    # Kept non-fatal: the log is a derived view, and losing it must never cost
    # the report itself.
    try:
        import build_forecast_log  # noqa: PLC0415
        build_forecast_log.build()
    except Exception as e:
        print(f"  WARNING: could not rebuild the forecast log ({type(e).__name__}: {e})")

    if args.no_send:
        print("\n--no-send, so here is the email that would go out:\n")
        print(body)
        return 0

    send_email(f"Surf forecast vs. actual — {target_date.strftime('%b %d, %Y')}",
               body, html=html_body,
               attachments=[out_path] + lookahead + [f[1] for f in frames] + args.attach)
    print("  Email sent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
