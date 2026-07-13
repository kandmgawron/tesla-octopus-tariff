#!/usr/bin/env python3
"""Detect Intelligent Octopus Go (IOG) free charging slots outside offpeak
and set Tesla Powerwall backup reserve to 100% to force grid charging.

Polls Octopus GraphQL API for planned dispatches. When a dispatch is active
outside the standard offpeak window (23:30-05:30), sets Powerwall backup
reserve to 100%. Restores the previous reserve when the dispatch ends.

Usage:
    python iog_powerwall_charger.py --octopus-api-key sk_live_xxx --octopus-account A-1234ABCD --tesla-email you@example.com

Or set environment variables:
    OCTOPUS_API_KEY, OCTOPUS_ACCOUNT, TESLA_EMAIL

Runs as a long-lived daemon polling every 2 minutes.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore

LOCAL_TZ = ZoneInfo("Europe/London")
GRAPHQL_URL = "https://api.octopus.energy/v1/graphql/"
OFFPEAK_START = (23, 30)  # HH, MM
OFFPEAK_END = (5, 30)
POLL_INTERVAL_SECONDS = 120
STATE_FILE = Path(".iog_powerwall_state.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Octopus GraphQL ──────────────────────────────────────────────────────────


def octopus_get_token(api_key: str) -> str:
    """Exchange Octopus API key for a JWT token."""
    query = """
    mutation APIKeyAuthentication($apiKey: String!) {
        obtainKrakenToken(input: {APIKey: $apiKey}) {
            token
        }
    }
    """
    resp = requests.post(
        GRAPHQL_URL,
        json={"query": query, "variables": {"apiKey": api_key}},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    token = data.get("data", {}).get("obtainKrakenToken", {}).get("token")
    if not token:
        raise RuntimeError(f"Octopus auth failed: {data}")
    return token


def octopus_get_dispatches(token: str, account_number: str) -> list[dict]:
    """Fetch planned dispatches from the Octopus GraphQL API."""
    query = """
    query getData($accountNumber: String!) {
        plannedDispatches(accountNumber: $accountNumber) {
            startDt
            endDt
            deltaKwh
            delta
            meta {
                source
                location
            }
        }
    }
    """
    resp = requests.post(
        GRAPHQL_URL,
        json={
            "query": query,
            "variables": {"accountNumber": account_number},
            "operationName": "getData",
        },
        headers={"Authorization": token},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    dispatches = data.get("data", {}).get("plannedDispatches")
    if dispatches is None:
        errors = data.get("errors", [])
        raise RuntimeError(f"Failed to fetch dispatches: {errors}")
    return dispatches


# ── Time helpers ─────────────────────────────────────────────────────────────


def is_in_offpeak(dt: datetime) -> bool:
    """Check if a datetime is within the standard IOG offpeak window (23:30-05:30 local)."""
    local = dt.astimezone(LOCAL_TZ)
    mins = local.hour * 60 + local.minute
    start_mins = OFFPEAK_START[0] * 60 + OFFPEAK_START[1]  # 23:30 = 1410
    end_mins = OFFPEAK_END[0] * 60 + OFFPEAK_END[1]  # 05:30 = 330
    # Window wraps midnight
    return mins >= start_mins or mins < end_mins


def get_active_outside_offpeak_dispatches(dispatches: list[dict]) -> list[dict]:
    """Return dispatches that are currently active AND outside the offpeak window."""
    now = datetime.now(timezone.utc)
    active = []
    for d in dispatches:
        start = datetime.fromisoformat(d["startDt"])
        end = datetime.fromisoformat(d["endDt"])
        if start <= now < end and not is_in_offpeak(now):
            active.append(d)
    return active


# ── Tesla Powerwall control ──────────────────────────────────────────────────


def get_tesla_battery(email: str):
    """Return the first Battery product from the Tesla account using teslapy."""
    import teslapy

    tesla = teslapy.Tesla(email, retry=2, timeout=30)
    tesla.redirect_uri = "tesla://auth/callback"

    if not tesla.authorized:
        log.error("Tesla not authorized. Run the main tariff tool's download-data command first to authenticate.")
        sys.exit(1)

    batteries = tesla.battery_list()
    if not batteries:
        log.error("No Powerwall found on Tesla account.")
        sys.exit(1)

    return batteries[0]


def set_powerwall_backup_reserve(battery, percent: int) -> None:
    """Set the Powerwall backup reserve percentage."""
    log.info(f"Setting Powerwall backup reserve to {percent}%")
    battery.set_backup_reserve_percent(percent)


# ── State persistence ────────────────────────────────────────────────────────


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ── Main loop ────────────────────────────────────────────────────────────────


def run_once(
    octopus_api_key: str,
    octopus_account: str,
    tesla_email: str,
    normal_reserve: int,
    dry_run: bool = False,
) -> None:
    """Single poll iteration."""
    state = load_state()

    # Get Octopus dispatches
    token = octopus_get_token(octopus_api_key)
    dispatches = octopus_get_dispatches(token, octopus_account)

    active_slots = get_active_outside_offpeak_dispatches(dispatches)

    if active_slots:
        slot = active_slots[0]
        end_dt = datetime.fromisoformat(slot["endDt"]).astimezone(LOCAL_TZ)
        log.info(
            f"Free IOG slot active! Ends at {end_dt.strftime('%H:%M')}. "
            f"Delta: {slot.get('deltaKwh', '?')} kWh"
        )

        if not state.get("charging_active"):
            # Store the current reserve so we can restore it later
            if not dry_run:
                battery = get_tesla_battery(tesla_email)
                # ponytail: we don't read current reserve from API because teslapy
                # doesn't expose it easily; we use the configured normal_reserve instead.
                set_powerwall_backup_reserve(battery, 100)
            state["charging_active"] = True
            state["previous_reserve"] = normal_reserve
            state["slot_end"] = slot["endDt"]
            save_state(state)
            log.info("Powerwall set to charge (backup reserve 100%)")
        else:
            log.info("Already charging — no action needed")

    else:
        if state.get("charging_active"):
            restore_to = state.get("previous_reserve", normal_reserve)
            log.info(f"No active free slot. Restoring backup reserve to {restore_to}%")
            if not dry_run:
                battery = get_tesla_battery(tesla_email)
                set_powerwall_backup_reserve(battery, restore_to)
            state["charging_active"] = False
            state.pop("slot_end", None)
            save_state(state)
        else:
            # Log upcoming dispatches outside offpeak for visibility
            now = datetime.now(timezone.utc)
            upcoming = [
                d for d in dispatches
                if datetime.fromisoformat(d["startDt"]) > now
                and not is_in_offpeak(datetime.fromisoformat(d["startDt"]))
            ]
            if upcoming:
                next_slot = min(upcoming, key=lambda d: d["startDt"])
                start_local = datetime.fromisoformat(next_slot["startDt"]).astimezone(LOCAL_TZ)
                log.info(f"Next free slot at {start_local.strftime('%H:%M %d/%m')}")
            else:
                log.info("No free slots outside offpeak scheduled")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect IOG free charging slots and charge Powerwall"
    )
    parser.add_argument(
        "--octopus-api-key",
        default=os.environ.get("OCTOPUS_API_KEY"),
        help="Octopus Energy API key (or OCTOPUS_API_KEY env var)",
    )
    parser.add_argument(
        "--octopus-account",
        default=os.environ.get("OCTOPUS_ACCOUNT"),
        help="Octopus account number e.g. A-1234ABCD (or OCTOPUS_ACCOUNT env var)",
    )
    parser.add_argument(
        "--tesla-email",
        default=os.environ.get("TESLA_EMAIL"),
        help="Tesla account email (or TESLA_EMAIL env var)",
    )
    parser.add_argument(
        "--normal-reserve",
        type=int,
        default=int(os.environ.get("POWERWALL_NORMAL_RESERVE", "20")),
        help="Normal backup reserve %% to restore after slot ends (default: 20)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=POLL_INTERVAL_SECONDS,
        help=f"Seconds between polls (default: {POLL_INTERVAL_SECONDS})",
    )
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually control Powerwall")
    args = parser.parse_args()

    if not args.octopus_api_key:
        parser.error("--octopus-api-key or OCTOPUS_API_KEY required")
    if not args.octopus_account:
        parser.error("--octopus-account or OCTOPUS_ACCOUNT required")
    if not args.tesla_email:
        parser.error("--tesla-email or TESLA_EMAIL required")

    log.info("IOG Powerwall Charger starting")
    log.info(f"  Account: {args.octopus_account}")
    log.info(f"  Tesla: {args.tesla_email}")
    log.info(f"  Normal reserve: {args.normal_reserve}%")
    log.info(f"  Poll interval: {args.poll_interval}s")
    if args.dry_run:
        log.info("  DRY RUN — Powerwall will not be controlled")

    # Graceful shutdown
    running = [True]

    def _stop(signum, frame):
        log.info("Shutting down...")
        running[0] = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    while running[0]:
        try:
            run_once(
                args.octopus_api_key,
                args.octopus_account,
                args.tesla_email,
                args.normal_reserve,
                dry_run=args.dry_run,
            )
        except Exception:
            log.exception("Error during poll cycle")

        if args.once:
            break
        time.sleep(args.poll_interval)

    return 0


if __name__ == "__main__":
    sys.exit(main())
