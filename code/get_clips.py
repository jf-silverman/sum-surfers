import os
import random
import sys
import time as time_mod
import requests
from pathlib import Path
from datetime import datetime, timedelta, time
from astral import LocationInfo
from astral.sun import sun
import pytz

# ---------- CONFIG ----------
CAMERA_ID = os.environ["SURFLINE_CAMERA_ID"]
ACCESS_TOKEN = os.environ["SURFLINE_ACCESS_TOKEN"]
# Resolve data dir relative to project root (parent of this script's directory)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = _PROJECT_ROOT / "data" / "not_needed_in_repo" / "surf_clips"
CLIP_DURATION_SEC = 5  # 5-second clips (standing default — see resolve_clip_duration)
# One-shot override: if this file exists, its contents (an integer, seconds) are used
# as the clip duration for THIS run only, then the file is deleted immediately so the
# override can never silently persist across runs — the next run always reverts to
# CLIP_DURATION_SEC above regardless of what happens during this run. Used for
# occasional one-off experiments (e.g. testing whether longer clips reduce frame-edge/
# wave-occlusion count variability) without risking eating into clip storage long-term.
CLIP_DURATION_OVERRIDE_FILE = _PROJECT_ROOT / "data" / ".clip_duration_override"
DAY_LOOKBACK_DAYS = int(os.environ.get("CLIP_LOOKBACK_DAYS", "5"))
REQUEST_TIMEOUT_SEC = float(os.environ.get("CLIP_REQUEST_TIMEOUT_SEC", "20"))
REQUEST_BASE_DELAY_SEC = float(os.environ.get("CLIP_REQUEST_DELAY_SEC", "0.35"))
REQUEST_JITTER_SEC = float(os.environ.get("CLIP_REQUEST_JITTER_SEC", "0.25"))

# Camera location — Pleasure Point, Santa Cruz, CA (confirmed 2026-08-28 via web
# search; the previous hardcoded coordinates, 33.790/-118.486, were wrong — that's
# Southern California, ~340 miles off. The error was silent because timezone
# (America/Los_Angeles) happened to be correct either way, and the resulting
# sunrise/sunset were plausible-looking, just off by ~15-25 minutes depending on
# season — enough to silently truncate real evening/morning clip collection.
LOCATION = dict(
    name="PleasurePoint",
    region="CA",
    timezone="America/Los_Angeles",
    latitude=36.9577,
    longitude=-121.9688,
)
# --------------------------------

BASE_URL = f"https://services.surfline.com/cameras/{CAMERA_ID}/clip?accessToken={ACCESS_TOKEN}"

# Browser-like headers required to pass Cloudflare's bot check on services.surfline.com
# (plain requests without these get a Cloudflare 403 challenge page regardless of token validity).
REQUEST_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Content-Type": "application/json",
    "Origin": "https://www.surfline.com",
    "Referer": "https://www.surfline.com/",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.5 Safari/605.1.15",
}


class AuthError(RuntimeError):
    """Raised specifically for a 401 from the clip JSON API — distinguishes an
    expired/invalid SURFLINE_ACCESS_TOKEN from other (possibly transient) failures,
    so main() can alert on it specifically instead of just logging and moving on."""


def _is_transient_response(status_code, body_text):
    if status_code in {429, 500, 502, 503, 504}:
        return True
    body = (body_text or "").lower()
    transient_markers = [
        "bad gateway",
        "cloudflare",
        "temporarily unavailable",
        "gateway timeout",
    ]
    return any(marker in body for marker in transient_markers)

# Surfline clip pattern (~9-minute intervals starting at 12:08 AM)
def generate_clip_start_times():
    times = []
    t = time(0, 8)
    while t < time(23, 59):
        times.append(t)
        total_minutes = t.hour * 60 + t.minute + 9
        t = time(total_minutes // 60 % 24, total_minutes % 60)
    return times

def find_nearest_clip_window(target_dt, clip_times):
    """Return datetime of nearest Surfline clip window ≤ target_dt"""
    for ct in reversed(clip_times):
        clip_dt = target_dt.replace(hour=ct.hour, minute=ct.minute, second=0, microsecond=0)
        if clip_dt <= target_dt:
            return clip_dt
    # fallback: first clip of the day
    return target_dt.replace(hour=clip_times[0].hour, minute=clip_times[0].minute, second=0, microsecond=0)

def download_clip(start_ms, end_ms, out_path):
    """Two-step API: request clip JSON, then download MP4"""
    payload = {"startTimestampInMs": start_ms, "endTimestampInMs": end_ms}

    try:
        resp = requests.post(BASE_URL, headers=REQUEST_HEADERS, json=payload, timeout=REQUEST_TIMEOUT_SEC)
    except requests.RequestException as e:
        raise RuntimeError(f"Transient network error requesting clip JSON: {type(e).__name__}: {e}") from e

    if resp.status_code != 200:
        body_preview = " ".join(resp.text.split())[:180]
        if resp.status_code == 401:
            raise AuthError(f"HTTP 401 from clip JSON API (SURFLINE_ACCESS_TOKEN expired/invalid). Body: {body_preview}")
        transient = _is_transient_response(resp.status_code, resp.text)
        label = "TRANSIENT" if transient else "NON_TRANSIENT"
        raise RuntimeError(
            f"{label} HTTP {resp.status_code} from clip JSON API. Body: {body_preview}"
        )

    try:
        resp_json = resp.json()
    except ValueError as e:
        raise RuntimeError("NON_TRANSIENT: Clip JSON API returned non-JSON response") from e

    clip_url = resp_json.get("clipUrl")
    if not clip_url:
        raise RuntimeError(f"NON_TRANSIENT: No clipUrl returned. Response: {resp_json}")

    try:
        r = requests.get(clip_url, stream=True, timeout=REQUEST_TIMEOUT_SEC)
        r.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Transient error downloading MP4 from clipUrl: {type(e).__name__}: {e}") from e

    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"💾 Saved {out_path.name}")

def send_auth_failure_email(auth_failure_count):
    """Non-fatal: email failure must never abort the pipeline (same pattern as
    manage_clips.py's storage-limit warning)."""
    try:
        sys.path.insert(0, str(_PROJECT_ROOT / "code"))
        from send_email import send_email  # noqa: PLC0415

        subject = "sum-surfers: Surfline access token invalid"
        body = (
            f"{auth_failure_count} clip download(s) failed today with HTTP 401 "
            "(Invalid Authentication) — SURFLINE_ACCESS_TOKEN in .env has expired "
            "or is invalid, so no new clips are being downloaded.\n\n"
            "To fix: grab a fresh accessToken from Chrome DevTools on the camera's "
            "page — open the Network tab, trigger a clip view/download, and find a "
            "request to services.surfline.com/cameras/<camera-id>/clip?accessToken=... "
            "Copy that value into SURFLINE_ACCESS_TOKEN in .env.\n"
        )
        send_email(subject, body)
        print("Auth-failure warning email sent.")
    except Exception as exc:
        print(f"Could not send auth-failure warning email: {exc}")


def get_light_window(date, local_tz):
    """Returns (first_light, last_light) datetimes for the clip-collection window.

    Prefers Surfline's own `sunlight` forecast endpoint (dawn/dusk = civil twilight,
    the standard "usable light" boundary) for TODAY specifically — it's free/no-token
    for the live forward-looking window, and it's literally the same source Surfline
    itself uses, so it can't drift from the real spot the way an independently
    computed astronomical estimate could. Falls back to astral (using this file's
    LOCATION, now corrected to Pleasure Point's real coordinates — confirmed via web
    search 2026-08-28 to match Surfline's own dawn/dusk within 1-2 minutes) for any
    date astral can't reach live (i.e. everything in DAY_LOOKBACK_DAYS's backfill
    window, which is in the past and would otherwise need a premium session token —
    see backfill_historical_predictors.py) or if the live fetch fails for any reason
    (network issue, API change, etc. — this must never block the whole clip run).

    Previously this used sunrise-30min/sunset+30min as a stand-in for first/last
    light — a flawed heuristic on two counts: it assumed a fixed 30-minute margin
    is always right (real dusk is ~25 min later than that at summer solstice for
    this location), and it inherited whatever error was in the hardcoded
    coordinates. dawn/dusk (civil twilight) is the actual concept wanted here.
    """
    loc = LocationInfo(**LOCATION)
    astral_result = sun(loc.observer, date=date, tzinfo=local_tz)
    astral_window = (astral_result["dawn"], astral_result["dusk"])

    if date != datetime.now(local_tz).date():
        return astral_window

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import get_surf_predictors as sp  # noqa: PLC0415

        params = {"spotId": sp.SPOT_ID, "days": "1", "intervalHours": "1"}
        resp = sp.fetch("sunlight", params)
        day = resp[0]
        dawn = datetime.fromtimestamp(day["dawn"], tz=pytz.utc).astimezone(local_tz)
        dusk = datetime.fromtimestamp(day["dusk"], tz=pytz.utc).astimezone(local_tz)
        return dawn, dusk
    except Exception as e:
        print(f"  WARNING: live Surfline sunlight fetch failed ({e}), falling back to astral")
        return astral_window


def resolve_clip_duration():
    """Returns the clip duration (seconds) to use for this run. Consumes (deletes)
    CLIP_DURATION_OVERRIDE_FILE immediately if present, so the override is truly
    one-shot even if this run crashes partway through."""
    if CLIP_DURATION_OVERRIDE_FILE.exists():
        raw = CLIP_DURATION_OVERRIDE_FILE.read_text().strip()
        CLIP_DURATION_OVERRIDE_FILE.unlink()
        try:
            value = int(raw)
        except ValueError:
            print(f"WARNING: {CLIP_DURATION_OVERRIDE_FILE} had non-integer contents ({raw!r}), "
                  f"ignoring and using default {CLIP_DURATION_SEC}s")
            return CLIP_DURATION_SEC
        print(f"One-off clip duration override in effect: {value}s for this run only "
              f"(override file consumed — next run reverts to default {CLIP_DURATION_SEC}s)")
        return value
    return CLIP_DURATION_SEC


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().date()
    clip_times_pattern = generate_clip_start_times()
    local_tz = pytz.timezone(LOCATION["timezone"])
    auth_failures = 0
    clip_duration_sec = resolve_clip_duration()

    for delta in range(0, DAY_LOOKBACK_DAYS):
        date = today - timedelta(days=delta)
        date_str = date.strftime("%Y-%m-%d")
        day_folder = OUT_DIR / date_str
        day_folder.mkdir(parents=True, exist_ok=True)

        first_light, last_light = get_light_window(date, local_tz)
        clip_dt = find_nearest_clip_window(first_light, clip_times_pattern)

        while clip_dt <= last_light:
            time_str = clip_dt.strftime("%H_%M")
            clip_folder = day_folder / time_str
            clip_folder.mkdir(exist_ok=True)
            clip_file = clip_folder / "clip.mp4"

            if clip_file.exists():
                print(f"✅ Clip exists: {date_str} {time_str}")
            else:
                start_ms = int(clip_dt.timestamp() * 1000)
                end_ms = start_ms + clip_duration_sec * 1000
                try:
                    download_clip(start_ms, end_ms, clip_file)
                except AuthError as e:
                    auth_failures += 1
                    print(f"⚠️ Failed: {date_str} {time_str}: {e}")
                except Exception as e:
                    print(f"⚠️ Failed: {date_str} {time_str}: {e}")
                # Single-attempt mode with pacing to reduce request bursts.
                delay = max(REQUEST_BASE_DELAY_SEC + random.uniform(0.0, REQUEST_JITTER_SEC), 0.0)
                time_mod.sleep(delay)

            # next clip ≥ 1 hour after previous
            next_target = clip_dt + timedelta(hours=1)
            clip_dt = find_nearest_clip_window(next_target, clip_times_pattern)

    if auth_failures > 0:
        print(f"\n{auth_failures} clip download(s) failed due to an invalid/expired access token.")
        send_auth_failure_email(auth_failures)

if __name__ == "__main__":
    main()