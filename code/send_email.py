#!/usr/bin/env python3
"""
send_email.py — Send a plain-text email via Gmail SMTP.

Credentials are read from environment variables (never hardcoded):
  SMTP_USER          Gmail address used to send (e.g. you@gmail.com)
  SMTP_APP_PASSWORD  Gmail App Password (not your regular password)
  EMAIL_TO           Recipient address (defaults to SMTP_USER if unset)

Usage as a script:
  python code/send_email.py --subject "Hello" --body "World"

Usage as a module:
  from send_email import send_email
  send_email("Hello", "World")
  send_email("Report", "See attached", attachments=["chart.png"])
"""

import argparse
import mimetypes
import os
import smtplib
import sys
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email import encoders
from pathlib import Path


def send_email(subject: str, body: str, attachments=None) -> None:
    """Send an email using credentials from env vars, optionally with files attached.

    `attachments` is a list of paths. A missing file raises rather than sending a
    report with its evidence silently absent — a daily report whose chart quietly
    failed to attach looks the same as one with nothing to say.
    """
    smtp_user = os.environ.get("SMTP_USER", "").strip()
    smtp_password = os.environ.get("SMTP_APP_PASSWORD", "").strip()
    email_to = os.environ.get("EMAIL_TO", smtp_user).strip()

    if not smtp_user or not smtp_password:
        raise EnvironmentError(
            "SMTP_USER and SMTP_APP_PASSWORD must be set in the environment."
        )
    if not email_to:
        raise EnvironmentError("EMAIL_TO (or SMTP_USER as fallback) must be set.")

    msg = MIMEMultipart()
    msg["From"] = smtp_user
    msg["To"] = email_to
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    for path in (attachments or []):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Attachment not found: {path}")
        ctype, _encoding = mimetypes.guess_type(path.name)
        maintype, subtype = (ctype or "application/octet-stream").split("/", 1)
        part = MIMEBase(maintype, subtype)
        part.set_payload(path.read_bytes())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", "attachment", filename=path.name)
        msg.attach(part)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(smtp_user, smtp_password)
        server.sendmail(smtp_user, email_to, msg.as_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send an email via Gmail SMTP.")
    parser.add_argument("--subject", required=True, help="Email subject line.")
    parser.add_argument("--body", required=True, help="Email body (plain text).")
    parser.add_argument("--attach", action="append", default=[],
                        help="Path to attach; repeatable.")
    args = parser.parse_args()

    try:
        send_email(args.subject, args.body, attachments=args.attach)
        print("Email sent.")
    except smtplib.SMTPException as e:
        print(f"SMTP error: {e}", file=sys.stderr)
        sys.exit(1)
    except EnvironmentError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)
