# GCP Setup Guide — Sum Surfers Pipeline

## Overview

The pipeline runs on a small GCP Compute Engine VM that wakes up at 20:00 PT every 3 days,
downloads clips, extracts frames, runs YOLOv8 inference, and then shuts itself down.
Each run takes about ~10 minutes. At spot pricing, compute cost is very low.

**Current architecture (hybrid):**
Surfline blocks GCP egress IPs, so clips must be downloaded from your laptop.

```
Laptop (cron, every 3 days)          GCP VM (Cloud Scheduler, every 3 days)
  1. get_clips.py                      ↓ starts on schedule (backup reminder)
  2. get_cropped_frame.py              vm_pipeline.sh checks for new crops
  3. Start VM                          → if none: emails reminder
  4. Upload crops to VM                → if found: runs detection
  5. SSH → vm_pipeline.sh              (rare; laptop usually triggers this)
  6. Pull predictions.csv back
  7. Stop VM
```

The VM's Cloud Scheduler job is a **backup reminder only** — it emails you if
the laptop cron job hasn't run recently. Normally the laptop triggers the VM
directly via SSH and pulls predictions before shutting the VM down.

---

## Step 1 — Install and authenticate the gcloud CLI (on your Mac)

```bash
brew install --cask google-cloud-sdk
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

---

## Step 2 — Create the VM

```bash
gcloud compute instances create surf-detector \
  --zone=us-west2-a \
  --machine-type=e2-standard-2 \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --boot-disk-size=50GB \
  --boot-disk-type=pd-balanced \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --scopes=cloud-platform \
  --tags=surf-detector
```

> **Why `e2-standard-2`?** 2 vCPU / 8 GB RAM is comfortable for YOLOv8s inference
> on 14 small images. Upgrade to `n1-standard-4` + T4 GPU only if inference is slow.

> **Spot VM** means GCP can preempt the VM when capacity is tight — but it will
> also start again on schedule.  Since the pipeline runs at night this is very
> rarely an issue.

---

## Step 3 — Copy the repo and model weights to the VM

```bash
# First time: clone the repo on the VM
gcloud compute ssh surf-detector --zone=us-west2-a -- \
  "git clone https://github.com/YOUR_USERNAME/sum-surfers.git ~/sum-surfers"

# Upload model weights from your Mac
gcloud compute scp \
  data/model_out/20251013/train/runs/detect/train13/weights/best.pt \
  surf-detector:~/sum-surfers/data/model_out/20251013/train/runs/detect/train13/weights/best.pt \
  --zone=us-west2-a

# Upload your .env file (contains secrets — do not commit to git)
gcloud compute scp .env surf-detector:~/sum-surfers/.env --zone=us-west2-a
```

---

## Step 4 — Run the one-time VM setup script

```bash
gcloud compute ssh surf-detector --zone=us-west2-a
# Inside the VM:
cd ~/sum-surfers
bash setup_vm.sh
```

Test the pipeline end-to-end before setting up the scheduler:
```bash
cd ~/sum-surfers
bash run_pipeline.sh
```

Also add the new email/reminder env vars to **the VM's `.env`** (SSH in and edit it):
```bash
gcloud compute ssh surf-detector --zone=us-west2-a -- \
  "cat >> ~/sum-surfers/.env" << 'EOF'
SMTP_USER=your_gmail_address@gmail.com
SMTP_APP_PASSWORD=your_16_char_app_password
EMAIL_TO=joelfsilverman@gmail.com
VM_DATA_LIMIT_GB=50
REMINDER_THRESHOLD_DAYS=4
EOF
```

---

## Step 4b — Set up the local cron job (on your Mac)

The local cron job pulls clips, crops frames, uploads to the VM, runs detection,
and pulls predictions back — all in one command.

**1. Make the script executable:**
```bash
chmod +x code/local_pipeline.sh
```

**2. Add email + GCP vars to your local `.env`** (copy from `.env.example` and fill in):
```
SMTP_USER=your_gmail_address@gmail.com
SMTP_APP_PASSWORD=your_16_char_app_password
EMAIL_TO=joelfsilverman@gmail.com
CLIPS_DIR_LIMIT_GB=1.0
GCP_PROJECT=sum-surfers-20260510-a1b2
GCP_ZONE=us-west2-a
GCP_INSTANCE=surf-detector
```

**3. Get your Gmail App Password:**
- Go to <https://myaccount.google.com/apppasswords>
- Create a new App Password (requires 2-Step Verification to be enabled)
- Paste the 16-character password as `SMTP_APP_PASSWORD` in your `.env`

**4. Edit your crontab:**
```bash
crontab -e
```
Add this line (adjust the path and time as needed; 06:00 daily every 3 days):
```
0 6 */3 * * /Users/YOUR_USERNAME/Documents/DS/sum-surfers/code/local_pipeline.sh >> /Users/YOUR_USERNAME/Documents/DS/sum-surfers/data/local_pipeline.log 2>&1
```

> **macOS note:** cron requires Full Disk Access.  Go to
> System Settings → Privacy & Security → Full Disk Access and add `/usr/sbin/cron`.

**5. Test a manual run:**
```bash
bash code/local_pipeline.sh
```

---

## Step 5 — Auto-start with Cloud Scheduler + Cloud Functions

GCP has no native "start a VM on schedule" feature, but this two-resource pattern
is the standard approach.

> **Role of Cloud Scheduler in the hybrid setup:** The scheduler is a *backup*
> reminder mechanism. It starts the VM every 3 days; `vm_pipeline.sh` runs,
> finds no new crops (if your laptop ran on schedule), and shuts back down
> silently. If your laptop *missed* its cron job (≥ 4 days since last run),
> the VM emails you a reminder instead.

### 5a — Enable APIs

```bash
gcloud services enable \
  cloudfunctions.googleapis.com \
  cloudscheduler.googleapis.com \
  compute.googleapis.com
```

### 5b — Create a Cloud Function to start the VM

```bash
mkdir /tmp/start-vm-fn && cat > /tmp/start-vm-fn/main.py << 'EOF'
import googleapiclient.discovery

def start_vm(request):
    service = googleapiclient.discovery.build("compute", "v1")
    service.instances().start(
        project="YOUR_PROJECT_ID",
        zone="us-west2-a",
        instance="surf-detector",
    ).execute()
    return "VM start requested", 200
EOF

cat > /tmp/start-vm-fn/requirements.txt << 'EOF'
google-api-python-client
EOF

gcloud functions deploy start-surf-detector \
  --gen2 \
  --runtime=python311 \
  --region=us-west2 \
  --source=/tmp/start-vm-fn \
  --entry-point=start_vm \
  --trigger-http \
  --no-allow-unauthenticated \
  --service-account=YOUR_SA_EMAIL
```

### 5c — Grant the function's service account permission to start VMs

```bash
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:YOUR_SA_EMAIL" \
  --role="roles/compute.instanceAdmin.v1"
```

### 5d — Create the Cloud Scheduler job (triggers at 19:55 PT every 3 days)

```bash
gcloud scheduler jobs create http start-surf-detector-every-3-days \
  --location=us-west2 \
  --schedule="55 19 */3 * *" \
  --time-zone="America/Los_Angeles" \
  --uri="https://us-west2-YOUR_PROJECT_ID.cloudfunctions.net/start-surf-detector" \
  --http-method=POST \
  --oidc-service-account-email=YOUR_SA_EMAIL
```

The VM starts at 19:55, cron fires `run_pipeline.sh` at 20:00, the pipeline
finishes ~20:10, and `sudo shutdown -h now` turns the VM off.

The VM's **startup script** (set once via the Cloud Console or gcloud) should call
`vm_pipeline.sh` so it acts as the backup reminder check:
```bash
# Set the VM startup script (run once from your Mac)
gcloud compute instances add-metadata surf-detector \
  --zone=us-west2-a \
  --metadata=startup-script='#!/bin/bash
cd /home/surfer/sum-surfers
source .venv/bin/activate
bash code/vm_pipeline.sh >> /var/log/surfers.log 2>&1'
```

---

## Step 6 — Retrieve predictions from the VM (optional)

After the pipeline runs, sync the predictions CSV back to your Mac or to GCS:

```bash
# Sync from VM to your Mac
gcloud compute scp \
  surf-detector:~/sum-surfers/data/predictions.csv \
  data/predictions.csv \
  --zone=us-west2-a

# OR — sync to a GCS bucket (add this to run_pipeline.sh before shutdown)
# gsutil cp "$PROJECT_DIR/data/predictions.csv" gs://YOUR_BUCKET/predictions.csv
```

> **In the hybrid setup, `local_pipeline.sh` does this automatically** (Step 8 of
> the local script). You only need the command above for one-off manual syncs.

---

## Secrets — Production Hardening (recommended)

Instead of a plaintext `.env` file, store secrets in **Secret Manager**:

```bash
# Store secrets once
echo -n "YOUR_CAMERA_ID"    | gcloud secrets create SURFLINE_CAMERA_ID   --data-file=-
echo -n "YOUR_ACCESS_TOKEN" | gcloud secrets create SURFLINE_ACCESS_TOKEN --data-file=-

# Read them at runtime in run_pipeline.sh (replace the .env sourcing block with):
# export SURFLINE_CAMERA_ID=$(gcloud secrets versions access latest --secret=SURFLINE_CAMERA_ID)
# export SURFLINE_ACCESS_TOKEN=$(gcloud secrets versions access latest --secret=SURFLINE_ACCESS_TOKEN)
```

Grant the VM's default service account access:
```bash
gcloud secrets add-iam-policy-binding SURFLINE_CAMERA_ID \
  --member="serviceAccount:$(gcloud compute instances describe surf-detector \
      --zone=us-west2-a --format='get(serviceAccounts[0].email)')" \
  --role="roles/secretmanager.secretAccessor"
```

---

## Cost Estimate

| Resource | Usage | Est. monthly cost |
|---|---|---|
| e2-standard-2 spot VM | ~10 min/day | < $0.10 |
| 50 GB pd-balanced disk | always-on | ~$4.50 |
| Cloud Scheduler | 1 job | free tier |
| Cloud Functions | 1 invocation/day | free tier |
| Egress (clips download from Surfline) | ~70 MB/day | negligible |
| **Total** | | **~$5/month** |
