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


def extreme_frames(target_date, model=None):
    """Detector frames for the day's busiest hour, an empty hour, and the quietest
    non-empty hour — the three points where counting is hardest to trust.

    Returns [(label, path, count), ...]; an empty hour is included only if the
    day had one.
    """
    df = pd.read_csv(PREDICTIONS_CSV, float_precision="round_trip")
    df = df[(df["date"] == target_date.isoformat())
            & (df["quality_ok"].astype(str) == "True")
            & df["surfer_count"].notna()]
    if df.empty:
        return []

    picks = []
    busiest = df.loc[df["surfer_count"].idxmax()]
    picks.append(("Busiest hour", busiest))
    zeros = df[df["surfer_count"] == 0]
    if not zeros.empty:
        picks.append(("Empty hour", zeros.iloc[0]))
    nonzero = df[df["surfer_count"] > 0]
    if not nonzero.empty:
        quietest = nonzero.loc[nonzero["surfer_count"].idxmin()]
        if quietest["filename"] != busiest["filename"]:
            picks.append(("Quietest hour above zero", quietest))

    if model is None:
        model = pdp.ds.load_model()

    out = []
    for label, row in picks:
        img_path = pdp.ds.CROPS_DIR / row["filename"]
        if not img_path.exists():
            continue
        frame, visible = pdp.render_detection_frame(img_path, model)
        if frame is None:
            continue
        banner_h = 44
        h, w = frame.shape[:2]
        canvas = np.zeros((h + banner_h, w, 3), dtype=np.uint8)
        canvas[:h] = frame
        hh, mm = map(int, str(row["time_local"]).split(":")[:2])
        stamp = datetime(target_date.year, target_date.month, target_date.day,
                         hh, mm).strftime("%-I:%M %p")
        cv2.putText(canvas, f"{label}  |  {stamp}  |  {visible} detected",
                    (10, h + 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (235, 235, 235), 2,
                    cv2.LINE_AA)
        slug = label.lower().split()[0]
        out_path = OUT_DIR / f"frame_{target_date.isoformat()}_{slug}.png"
        cv2.imwrite(str(out_path), canvas)
        out.append((label, out_path, visible, stamp))
    return out


def build_chart(target_date, forecast_rows, actuals, recorded_at, out_path, tide=None):
    fig, ax = plt.subplots(figsize=(13, 6.2), facecolor=pdp.BG_COLOR)
    ax.set_facecolor(pdp.AXES_BG)

    fh = [r["hour"] for r in forecast_rows]
    lower = [r["lower"] for r in forecast_rows]
    upper = [r["upper"] for r in forecast_rows]
    point = [r["point"] for r in forecast_rows]

    ax.fill_between(fh, lower, upper, color=pdp.AQUA, alpha=0.22, linewidth=0,
                    label="Forecast 80% range")
    ax.plot(fh, point, color=pdp.AQUA, linewidth=2.5, label="Forecast")

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

    provenance = (f"Forecast recorded {recorded_at}" if recorded_at
                  else "Forecast RECONSTRUCTED after the fact (no record existed for this "
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
    if recorded:
        forecast_rows, made_at = recorded
        print(f"  Using the recorded forecast (made {made_at})")
    elif args.allow_reconstruct:
        print("  No recorded forecast for this date — reconstructing from rows before it.")
        forecast_rows, made_at = reconstruct_forecast(target_date)
    else:
        print("  No recorded forecast and reconstruction disabled — nothing to send.")
        return 1

    tide = tide_by_hour(target_date)
    out_path = OUT_DIR / f"forecast_vs_actual_{target_date.isoformat()}.png"
    build_chart(target_date, forecast_rows, actuals, made_at, out_path, tide=tide)
    print(f"  Chart: {out_path}  (tide points: {len(tide)})")

    frames = extreme_frames(target_date)
    for label, fpath, count, stamp in frames:
        print(f"  {label}: {stamp}, {count} detected -> {fpath.name}")

    table, stats = summarize(forecast_rows, actuals)
    provenance = (f"Forecast recorded {made_at}." if made_at else
                  "NOTE: no forecast was recorded for this date, so the forecast shown was "
                  "reconstructed afterwards by refitting on data from before this day. It is "
                  "a re-enactment, not the original forecast. Days from here on will use the "
                  "real recorded forecast.")
    headline = (f"Average miss {stats['mae']:.1f} surfers, bias {stats['bias']:+.1f} "
                f"({'over' if stats['bias'] > 0 else 'under'}-forecast), "
                f"{stats['inside']} of {stats['n']} hours inside the 80% range."
                if stats else "No overlapping hours to score.")

    if frames:
        frame_lines = "\n".join(
            f"  {label}: {stamp}, {count} detected  ({fpath.name})"
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

    body = (f"Surf crowd forecast vs. actual for {target_date.strftime('%A, %B %d, %Y')}\n\n"
            f"{headline}\n"
            f"{tide_note}\n"
            f"Hour by hour:\n{table}\n\n"
            f"Detector frames attached, at the three points where counting is hardest\n"
            f"to trust:\n{frame_lines}\n\n"
            f"{provenance}\n\n"
            f"Chart attached, with tide on the right-hand axis. Counts come from the\n"
            f"detector, not a human, so they carry its own error (about 1 surfer per\n"
            f"frame on held-out images).\n")

    if args.no_send:
        print("\n--no-send, so here is the email that would go out:\n")
        print(body)
        return 0

    send_email(f"Surf forecast vs. actual — {target_date.strftime('%b %d, %Y')}",
               body, attachments=[out_path] + [f[1] for f in frames] + args.attach)
    print("  Email sent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
