#!/usr/bin/env python3
"""
send_email.py — Send a plain-text email via Gmail SMTP.

Credentials are read from environment variables (never hardcoded):
  SMTP_USER          Gmail address used to send  (e.g. you@gmail.com)
  SMTP_APP_PASSWORD  Gmail App Password (not your regular password)
  EMAIL_TO           Recipient address (defaults to SMTP_USER if unset)

Usage as a script:
  python code/send_email.py --subject "Hello" --body "World"

Usage as a module:
  from send_email import send_email
  send_email("Hello", "World")
"""

import argparse
import os
import smtplib
import sys
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def send_email(subject: str, body: str) -> None:
    """Send a plain-text email using credentials from env vars."""
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

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(smtp_user, smtp_password)
        server.sendmail(smtp_user, email_to, msg.as_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send an email via Gmail SMTP.")
    parser.add_argument("--subject", required=True, help="Email subject line.")
    parser.add_argument("--body", required=True, help="Email body (plain text).")
    args = parser.parse_args()

    try:
        send_email(args.subject, args.body)
        print("Email sent.")
    except EnvironmentError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)
    except smtplib.SMTPException as e:
        print(f"SMTP error: {e}", file=sys.stderr)
        sys.exit(1)
