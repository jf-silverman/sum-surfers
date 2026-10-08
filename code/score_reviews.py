"""
score_reviews.py
----------------
Scores the hand-filled review sheet at data/reviews/review_all.csv and
reports detector precision, recall and F1 -- as INTERVALS, not point estimates.

Why intervals. Each reviewed row carries two doubt columns, split by direction
(2026-10-05) because they bound different metrics:

    uncertain_box_numbers   drawn boxes the reviewer would not defend either
                            way. They might be true positives or false
                            positives, so they bound PRECISION.
    uncertain_missed        maybe-surfers with no box drawn. They might be real
                            misses or imagined, so they bound RECALL.

Both are a subset of the counts beside them: the reviewer still makes a best
call, then flags which calls are soft. So this reports a best case, a worst
case, and the reviewer's own best guess in between. The width between best and
worst is the ambiguity budget -- how much of the headline number is reviewer
judgement rather than measurement.

Bounding rule for uncertain boxes. The CSV records how many boxes are soft but
not whether each was called TP or FP, so the bound is taken without assuming:
best case moves every soft box to TP, worst case moves every soft box to FP,
both clamped to boxes_drawn. That is slightly wider than the truth whenever the
reviewer's calls were split, and never narrower -- the safe direction.

There is no true-negative here, and so no accuracy figure: an unbounded number
of image regions contain no surfer, and the detector is not asked about them.
Precision, recall and count error are the whole story.

Usage:
    python code/score_reviews.py
    python code/score_reviews.py --date 2026-09-23
    python code/score_reviews.py --by-day
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
REVIEWS = _PROJECT_ROOT / "data" / "reviews"
REVIEW_CSV = REVIEWS / "review_all.csv"

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

FILL_COLS = ["true_positives", "missed", "false_positives"]


def wilson(hits, n, z=1.96):
    """Wilson score interval -- behaves at the extremes where normal-approx does not."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = hits / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def n_listed(cell):
    """Count of ;-separated entries in a free-text list cell."""
    if not isinstance(cell, str) or not cell.strip():
        return 0
    return len([x for x in cell.replace(",", ";").split(";") if x.strip()])


def as_int(cell):
    if not isinstance(cell, str):
        cell = str(cell)
    cell = cell.strip()
    if not cell:
        return 0
    try:
        return int(float(cell))
    except ValueError:
        return 0


def load_rows(date_filter=None):
    """Reviewed rows only -- a row counts as reviewed once true_positives is filled."""
    if not REVIEW_CSV.exists():
        sys.exit(f"{REVIEW_CSV} not found.")
    rows, skipped, problems = [], 0, []
    df = pd.read_csv(REVIEW_CSV, dtype=str, keep_default_na=False)
    unusable = []
    for _, r in df.iterrows():
        day = str(r["date"]).strip()
        # Rows 2 and 3 are a column description and a worked example, for
        # whoever is filling the sheet in. A data row is identified by its date
        # LOOKING like a date -- deliberately not by a marker character. The
        # first attempt used a leading "#", which CSV editors treat as a comment
        # prefix and collapsed the whole row into one cell (2026-10-06).
        if not DATE_RE.match(day):
            continue
        if date_filter and day != date_filter:
            continue
        cell = str(r.get("true_positives", "")).strip()
        if not cell:
            skipped += 1
            continue
        # "na" = looked at and judged uncountable (noise, condensation, flare).
        # Kept apart from blank: blank means not yet reviewed, na is a verdict,
        # and a frame the quality gate passed but a human cannot count is a
        # finding about the gate rather than a gap in the review.
        if cell.lower() in ("na", "n/a"):
            unusable.append((day, r["time_local"], str(r.get("notes", "")).strip()))
            continue
        tp = as_int(r["true_positives"])
        # false_positives is DERIVED (2026-10-08). It used to be a stored count,
        # and across all 55 rows that had one it always equalled the number of
        # listed box numbers -- a duplicate that could only ever drift.
        fp = n_listed(r.get("false_positive_box_numbers", ""))
        drawn = as_int(r["boxes_drawn"])
        if tp + fp != drawn:
            problems.append(f"  {day} {r['time_local']}: "
                            f"true_positives({tp}) + false_positives({fp}) "
                            f"= {tp + fp}, but boxes_drawn = {drawn}")
        fn = as_int(r["missed"])
        # Each extra surfer inside a box costs one miss, so the parsed total
        # must not exceed `missed`.
        _, merged, bad_multi = parse_multi_surfer(r.get("multi_surfer_box", ""))
        for b_ in bad_multi:
            problems.append(f"  {day} {r['time_local']}: multi_surfer_box -- {b_}")
        if merged > fn:
            problems.append(f"  {day} {r['time_local']}: multi_surfer_box implies {merged} "
                            f"extra surfer(s) but missed = {fn}")
        rows.append(dict(
            date=day, time_local=r["time_local"], drawn=drawn, tp=tp, fp=fp,
            fn=fn, merged=min(merged, fn),
            standing=str(r.get("standing_box_numbers", "")).strip(),
            n_standing=n_listed(r.get("standing_box_numbers", "")),
            sup=str(r.get("sup_box_numbers", "")).strip(),
            n_sup=n_listed(r.get("sup_box_numbers", "")),
            dup_box=n_listed(r.get("multi_box_surfer", "")),
            u_box=n_listed(r.get("uncertain_box_numbers", "")),
            u_missed=as_int(r.get("uncertain_missed", "")),
            causes=r.get("false_positive_causes", ""),
            poses=r.get("missed_poses", "")))
    return pd.DataFrame(rows), skipped, problems, unusable


def score(df):
    """Point estimate plus best/worst bounds, aggregated over the rows given."""
    tp, fp, fn = int(df.tp.sum()), int(df.fp.sum()), int(df.fn.sum())
    drawn, u_box, u_missed = int(df.drawn.sum()), int(df.u_box.sum()), int(df.u_missed.sum())

    # Precision: soft boxes swing to TP (best) or to FP (worst), clamped.
    tp_best = min(tp + u_box, drawn)
    tp_worst = max(tp - u_box, 0)
    # Recall: soft misses are imagined (best) or real (worst). They are a subset
    # of fn, so the best case removes them and the worst case keeps them.
    fn_best = max(fn - u_missed, 0)

    def pr(t, f):
        return t / (t + f) if (t + f) else float("nan")

    out = dict(
        n_frames=len(df), tp=tp, fp=fp, fn=fn, drawn=drawn, merged=int(df.merged.sum()),
        dup_box=int(df.dup_box.sum()),
        u_box=u_box, u_missed=u_missed,
        precision=pr(tp, fp),
        precision_best=pr(tp_best, drawn - tp_best),
        precision_worst=pr(tp_worst, drawn - tp_worst),
        recall=tp / (tp + fn) if (tp + fn) else float("nan"),
        recall_best=tp_best / (tp_best + fn_best) if (tp_best + fn_best) else float("nan"),
        recall_worst=tp_worst / (tp_worst + fn) if (tp_worst + fn) else float("nan"),
    )
    p, r = out["precision"], out["recall"]
    out["f1"] = 2 * p * r / (p + r) if (p and r and not math.isnan(p + r)) else float("nan")
    out["precision_wilson"] = wilson(tp, tp + fp)
    out["recall_wilson"] = wilson(tp, tp + fn)

    # Count error: what the pipeline actually records vs what was really there.
    truth = df.tp + df.fn
    err = df.drawn - truth
    out["count_mae"] = float(err.abs().mean())
    out["count_bias"] = float(err.mean())
    out["true_total"] = int(truth.sum())
    return out


def fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


_MULTI_ENTRY = re.compile(r"^(\d+)(?:\(\s*(\d+)\s*\))?$")


def parse_multi_surfer(cell):
    """Parse the multi_surfer_box cell -> (entries, extra_surfers, bad).

    Syntax (Joel, 2026-10-06): box numbers separated by ";", with the surfer
    count in parentheses when it is more than the usual two.

        "2"            box 2 holds 2 surfers      -> 1 extra
        "2;3"          boxes 2 and 3 hold 2 each  -> 2 extra
        "2(3); 3"      box 2 holds 3, box 3 holds 2 -> 3 extra
        "2(3); 3(2)"   the same, written explicitly

    "Extra" means surfers with no box of their own, which is what has to appear
    in `missed`. A box always accounts for one surfer itself, so a box of N
    contributes N-1.

    The older form listed a box once per extra surfer ("1;1" for a box of 3).
    That still parses to the same total, so old rows need no migration, but a
    repeated box number is reported so it can be rewritten the clear way.
    """
    entries, extra, bad = [], 0, []
    for item in str(cell).replace(",", ";").split(";"):
        item = item.strip()
        if not item:
            continue
        m = _MULTI_ENTRY.match(item.replace(" ", ""))
        if not m:
            bad.append(item)
            continue
        box = int(m.group(1))
        n = int(m.group(2)) if m.group(2) else 2
        if n < 2:
            bad.append(f"{item} (a multi-surfer box needs 2 or more)")
            continue
        entries.append((box, n))
        extra += n - 1
    dupes = [b for b, _ in entries if [x for x, _ in entries].count(b) > 1]
    if dupes:
        bad.append(f"box {sorted(set(dupes))[0]} listed more than once -- "
                   f"write it as {sorted(set(dupes))[0]}(N) instead")
    return entries, extra, bad


_LEADING_COUNT = re.compile(r"^\s*(\d+)\s+(.*)$")


def tally_poses(df):
    """Count missed surfers by pose, honouring an inline leading count.

    Joel writes these three ways, all of which appear in 2026-09-23:
    "prone" (one, matching missed=1), "2 prone" (two), and "1 prone; 1 sitting"
    (one each). A naive split makes "2 prone" its own label and undercounts the
    real thing, so a leading integer is read as a multiplier.

    Returns the tally plus any rows whose poses do not add up to `missed`, which
    is usually a pose simply left blank rather than an error.
    """
    counts, unexplained = {}, []
    for _, r in df.iterrows():
        total = 0
        for item in str(r["poses"]).split(";"):
            item = item.strip().lower()
            if not item:
                continue
            m = _LEADING_COUNT.match(item)
            n, label = (int(m.group(1)), m.group(2).strip()) if m else (1, item)
            counts[label] = counts.get(label, 0) + n
            total += n
        if total != r["fn"]:
            unexplained.append((r["date"], r["time_local"], r["fn"], total))
    return sorted(counts.items(), key=lambda kv: -kv[1]), unexplained


def tally_causes(df):
    """Frames citing each false-positive cause.

    Deliberately counts FRAMES, not false positives. The cause text is free
    form and does not map one-to-one onto boxes: on 2026-09-23 one FP carried
    two semicolon-separated clauses ("lens flare spot; very distinct oversat
    ...") while elsewhere two FPs shared a single cause ("heads of people on
    shore"). Reporting these as per-box counts would be made up.
    """
    counts = {}
    for _, r in df.iterrows():
        cell = str(r["causes"]).strip()
        if not cell:
            continue
        for item in cell.split(";"):
            item = item.strip().lower()
            if item:
                counts[item] = counts.get(item, 0) + 1
    return sorted(counts.items(), key=lambda kv: -kv[1])


def report(s, title):
    print(f"\n{title}")
    print("-" * len(title))
    print(f"  frames reviewed      {s['n_frames']}")
    print(f"  boxes drawn          {s['drawn']}        true surfers  {s['true_total']}")
    print(f"  TP {s['tp']}   FP {s['fp']}   FN {s['fn']}")
    if s["fn"]:
        merged, nobox = s["merged"], s["fn"] - s["merged"]
        print(f"    of the {s['fn']} missed: {merged} inside a box the detector merged, "
              f"{nobox} with no box at all")
    if s["fp"]:
        dup, other = s["dup_box"], s["fp"] - s["dup_box"]
        print(f"    of the {s['fp']} false positives: {dup} extra box(es) on a surfer "
              f"already counted, {other} on something that is not a surfer")
    if s["u_box"] or s["u_missed"]:
        print(f"  flagged uncertain    {s['u_box']} box(es), {s['u_missed']} possible miss(es)")
    print()
    pw, rw = s["precision_wilson"], s["recall_wilson"]
    print(f"  precision   {fmt(s['precision'])}   "
          f"[{fmt(s['precision_worst'])} - {fmt(s['precision_best'])}] ambiguity   "
          f"({fmt(pw[0])}-{fmt(pw[1])}) Wilson 95%")
    print(f"  recall      {fmt(s['recall'])}   "
          f"[{fmt(s['recall_worst'])} - {fmt(s['recall_best'])}] ambiguity   "
          f"({fmt(rw[0])}-{fmt(rw[1])}) Wilson 95%")
    print(f"  F1          {fmt(s['f1'])}")
    print()
    print(f"  count MAE   {fmt(s['count_mae'], 2)} surfers/frame      "
          f"bias {s['count_bias']:+.2f} (negative = undercount)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date", help="Score one day only, YYYY-MM-DD")
    p.add_argument("--by-day", action="store_true", help="Also break the totals down per day")
    args = p.parse_args()

    df, skipped, problems, unusable = load_rows(args.date)
    if df.empty:
        sys.exit("No reviewed rows found. Fill in true_positives in "
                 f"{REVIEW_CSV} first.")

    if problems:
        print("ARITHMETIC PROBLEMS (true_positives + false_positives should equal boxes_drawn):")
        print("\n".join(problems))

    days = ", ".join(sorted(df.date.unique()))
    report(score(df), f"OVERALL  ({df.date.nunique()} day(s): {days})")

    if args.by_day and df.date.nunique() > 1:
        for day, sub in df.groupby("date"):
            report(score(sub), f"{day}")

    causes = tally_causes(df)
    if causes:
        print("\nFalse-positive causes, by frames citing them")
        print("--------------------------------------------")
        for name, n in causes:
            print(f"  {n:3d}  {name}")
        print("  (frames, not boxes -- the cause text does not map 1:1 onto FPs)")

    poses, unexplained = tally_poses(df)
    if poses:
        print("\nMissed surfers by pose")
        print("----------------------")
        for name, n in poses:
            print(f"  {n:3d}  {name}")
        print(f"  {sum(n for _, n in poses)} of {int(df.fn.sum())} missed surfers have a pose recorded")
        for date, t, fn, got in unexplained:
            print(f"    {date} {t}: missed={fn} but {got} pose(s) listed")

    # The two rare classes, reported separately because they are separate
    # classes: standing means riding a wave, SUP is a paddleboarder.
    # The two box-merging failures are one mechanism seen from both sides.
    mg, dup = int(df.merged.sum()), int(df.dup_box.sum())
    if mg or dup:
        print("\nBox merging, both directions (the same NMS / containment mechanism)")
        print("-" * 62)
        print(f"  {mg:3d} surfer(s) lost inside a box drawn around several  "
              f"-- suppression too aggressive")
        print(f"  {dup:3d} extra box(es) on one surfer                      "
              f"-- suppression not aggressive enough")

    for col, n_col, label, labeled in (("standing", "n_standing", "STANDING (riding a wave)", 31),
                                       ("sup", "n_sup", "SUP (stand-up paddleboard)", 4)):
        if n_col not in df.columns:
            continue
        hits = df[df[n_col] > 0]
        if not len(hits):
            continue
        total = int(hits[n_col].sum())
        print(f"\n{label}: {total} box(es) located across {len(hits)} frame(s)")
        print("-" * 62)
        for _, r in hits.sort_values(["date", "time_local"]).iterrows():
            print(f"  {r['date']} {r['time_local']}  box {r[col]}")
        print(f"  CVAT shortlist -- frames are on disk and the boxes are already drawn.")
        print(f"  The labeled training set holds {labeled} of these, against 617 sitting")
        print(f"  and 512 prone; that imbalance is what blocks the multi-class detector.")
    if unusable:
        print(f"\nUnusable frames ({len(unusable)}), marked na and excluded from every figure")
        print("-" * 62)
        for day, t, note in unusable:
            print(f"  {day} {t}  {note}")
        print("  These passed the automated quality gate but a human could not count them.")
    print(f"\n{skipped} row(s) not yet reviewed, skipped.")
    print("\nNo true-negative exists for a detector, so there is no accuracy figure:")
    print("the count of image regions correctly left un-boxed is unbounded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
