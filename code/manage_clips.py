#!/usr/bin/env python3
"""
manage_clips.py — Local clips storage checker and interactive cleaner.

Modes:
  python code/manage_clips.py           Interactive: check size and prompt if over limit.
  python code/manage_clips.py --check   Non-interactive: send a warning email if over limit.

Environment variables (read from .env or shell):
  CLIPS_DIR_LIMIT_GB   Storage limit in GB before warning (default: 1.0)
  SMTP_USER            For email warnings in --check mode
  SMTP_APP_PASSWORD    For email warnings in --check mode
  EMAIL_TO             Recipient (default: SMTP_USER)
"""

import argparse
import os
import shutil
import sys
from datetime import date, datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLIPS_DIR = _PROJECT_ROOT / "data" / "not_needed_in_repo" / "surf_clips"
LIMIT_GB = float(os.environ.get("CLIPS_DIR_LIMIT_GB", "1.0"))


def get_dir_size_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024**3)


def get_date_folders(clips_dir: Path) -> list[tuple[date, Path]]:
    """Return (date, path) pairs sorted oldest-first."""
    folders = []
    for entry in clips_dir.iterdir():
        if entry.is_dir():
            try:
                d = datetime.strptime(entry.name, "%Y-%m-%d").date()
                folders.append((d, entry))
            except ValueError:
                pass
    return sorted(folders, key=lambda x: x[0])


def measure_folders(folders: list[tuple[date, Path]]) -> float:
    total = sum(
        f.stat().st_size
        for _, path in folders
        for f in path.rglob("*")
        if f.is_file()
    )
    return total / (1024**3)


def delete_folders(folders: list[tuple[date, Path]]) -> tuple[int, float]:
    freed_gb = measure_folders(folders)
    for _, path in folders:
        shutil.rmtree(path)
    return len(folders), freed_gb


def run_interactive(clips_dir: Path, current_gb: float) -> None:
    print(f"\nClips directory : {clips_dir}")
    print(f"Current size    : {current_gb:.2f} GB  (limit: {LIMIT_GB:.1f} GB)")

    if current_gb <= LIMIT_GB:
        print("Storage is within limit. No action needed.")
        return

    folders = get_date_folders(clips_dir)
    if not folders:
        print("No date folders found in clips directory.")
        return

    oldest = folders[0][0]
    newest = folders[-1][0]
    print(f"\n{len(folders)} date folders spanning {oldest} → {newest}")
    print("\nDelete options (oldest clips removed first):")
    print("  1) Delete oldest month of clips")
    print("  2) Delete oldest 3 months of clips")
    print("  3) Keep most recent year only (delete everything older than 12 months)")
    print("  4) Cancel — do nothing")

    choice = input("\nEnter choice [1-4]: ").strip()

    today = date.today()

    if choice == "1":
        target_month = (oldest.year, oldest.month)
        to_delete = [(d, p) for d, p in folders if (d.year, d.month) == target_month]
    elif choice == "2":
        months: list[tuple[int, int]] = []
        for d, _ in folders:
            m = (d.year, d.month)
            if m not in months:
                months.append(m)
            if len(months) == 3:
                break
        to_delete = [(d, p) for d, p in folders if (d.year, d.month) in months]
    elif choice == "3":
        cutoff = date(today.year - 1, today.month, today.day)
        to_delete = [(d, p) for d, p in folders if d < cutoff]
    elif choice == "4":
        print("Cancelled.")
        return
    else:
        print("Invalid choice.")
        return

    if not to_delete:
        print("No folders match the selected criteria.")
        return

    span = f"{to_delete[0][0]} → {to_delete[-1][0]}"
    size_gb = measure_folders(to_delete)
    print(f"\nWill delete {len(to_delete)} folder(s) ({span}), ~{size_gb:.2f} GB")
    confirm = input("Confirm? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return

    count, freed = delete_folders(to_delete)
    new_size = get_dir_size_gb(clips_dir)
    print(f"Deleted {count} folder(s), freed {freed:.2f} GB. New size: {new_size:.2f} GB")


def run_check(clips_dir: Path, current_gb: float) -> None:
    """Non-interactive: send a warning email if over limit, then exit cleanly."""
    if current_gb <= LIMIT_GB:
        print(f"Clips storage OK: {current_gb:.2f} GB / {LIMIT_GB:.1f} GB")
        return

    print(f"WARNING: clips storage {current_gb:.2f} GB exceeds {LIMIT_GB:.1f} GB limit.")

    try:
        sys.path.insert(0, str(_PROJECT_ROOT / "code"))
        from send_email import send_email  # noqa: PLC0415

        subject = "sum-surfers: local clips storage limit exceeded"
        body = (
            f"Your local surf clips directory is {current_gb:.2f} GB, "
            f"which exceeds the {LIMIT_GB:.1f} GB limit.\n\n"
            f"Directory: {clips_dir}\n\n"
            "To free space, run the interactive cleaner:\n"
            "  cd /path/to/sum-surfers\n"
            "  source .env && python code/manage_clips.py\n\n"
            "Options offered:\n"
            "  1) Delete oldest month\n"
            "  2) Delete oldest 3 months\n"
            "  3) Keep most recent year only\n"
        )
        send_email(subject, body)
        print("Warning email sent.")
    except Exception as exc:
        # Email failure must not abort the pipeline
        print(f"Could not send warning email: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage local surf clips storage.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Non-interactive mode: email a warning if over limit.",
    )
    args = parser.parse_args()

    if not CLIPS_DIR.exists():
        print(f"Clips directory not found: {CLIPS_DIR} — skipping check.")
        return

    current_gb = get_dir_size_gb(CLIPS_DIR)

    if args.check:
        run_check(CLIPS_DIR, current_gb)
    else:
        run_interactive(CLIPS_DIR, current_gb)


if __name__ == "__main__":
    main()
