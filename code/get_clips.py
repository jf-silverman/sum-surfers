import requests
from pathlib import Path
from datetime import datetime, timedelta, time
from astral import LocationInfo
from astral.sun import sun
import pytz

# ---------- CONFIG ----------
CAMERA_ID = "5834946f3421b20545c4b51a"  # replace with your camera ID
ACCESS_TOKEN = "c3ab57d4bffd52ed467dcfa4b7a14d38d711a316"  # replace with your token
OUT_DIR = Path("data/not_needed_in_repo/surf_clips")
CLIP_DURATION_SEC = 5  # 5-second clips

# Camera location for sunrise/sunset
LOCATION = dict(
    name="SurfCam",
    region="CA",
    timezone="America/Los_Angeles",
    latitude=33.790,
    longitude=-118.486,
)
# --------------------------------

BASE_URL = f"https://services.surfline.com/cameras/{CAMERA_ID}/clip?accessToken={ACCESS_TOKEN}"

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
    headers = {"Content-Type": "application/json"}
    resp = requests.post(BASE_URL, headers=headers, json=payload)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to request clip JSON: {resp.status_code} {resp.text}")

    clip_url = resp.json().get("clipUrl")
    if not clip_url:
        raise RuntimeError(f"No clipUrl returned: {resp.text}")

    r = requests.get(clip_url, stream=True)
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"💾 Saved {out_path.name}")

def main():
    OUT_DIR.mkdir(exist_ok=True)
    today = datetime.now().date()
    clip_times_pattern = generate_clip_start_times()
    local_tz = pytz.timezone(LOCATION["timezone"])

    for delta in range(0, 5):  # today + prior 4 days
        date = today - timedelta(days=delta)
        date_str = date.strftime("%Y-%m-%d")
        day_folder = OUT_DIR / date_str
        day_folder.mkdir(parents=True, exist_ok=True)

        # sunrise/sunset
        loc = LocationInfo(**LOCATION)
        s = sun(loc.observer, date=date, tzinfo=local_tz)
        sunrise = s["sunrise"]
        sunset = s["sunset"]

        # first clip: includes 30 min before sunrise
        first_target = sunrise - timedelta(minutes=30)
        clip_dt = find_nearest_clip_window(first_target, clip_times_pattern)

        while clip_dt <= sunset + timedelta(minutes=30):
            time_str = clip_dt.strftime("%H_%M")
            clip_folder = day_folder / time_str
            clip_folder.mkdir(exist_ok=True)
            clip_file = clip_folder / "clip.mp4"

            if clip_file.exists():
                print(f"✅ Clip exists: {date_str} {time_str}")
            else:
                start_ms = int(clip_dt.timestamp() * 1000)
                end_ms = start_ms + CLIP_DURATION_SEC * 1000
                try:
                    download_clip(start_ms, end_ms, clip_file)
                except Exception as e:
                    print(f"⚠️ Failed: {date_str} {time_str}: {e}")

            # next clip ≥ 1 hour after previous
            next_target = clip_dt + timedelta(hours=1)
            clip_dt = find_nearest_clip_window(next_target, clip_times_pattern)

if __name__ == "__main__":
    main()