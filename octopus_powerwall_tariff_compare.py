#!/usr/bin/env python3
"""
octopus_powerwall_tariff_compare.py

Compare a Tesla Powerwall home's historical usage against Octopus tariffs using:
- Tesla Powerwall CSV exports (5-minute power data required)
- Optional Tesla account fetch via TeslaPy (unofficial owner API)
- Optional local gateway fetch via pyPowerwall (community library)

Key features
------------
- Compares historical costs under:
  * Intelligent-style tariff (fixed cheap window + fixed export)
  * Agile import + Agile outgoing from saved CSVs
- Models battery-limited cheap energy for Intelligent
- Models smart shifting for Agile by moving flexible demand into the cheapest
  half-hour slots of each day, subject to a configurable charge window/capacity
- Allows manual extra daily usage (e.g. sauna / hot tub), with time window
- Writes a daily audit CSV showing:
  * whether the battery ran out
  * cheap vs expensive kWh
  * flexible demand handled at cheap rates
  * direct import / export
- Can discover available Agile tariff CSVs by region from a folder

Important modelling notes
-------------------------
1. This script uses historical *grid import/export* from Tesla 5-minute power data.
2. For Intelligent modelling, cheap energy is capped by available daily battery
   storage. It is not reused more than once per day.
3. For Agile modelling, flexible demand is shifted into the cheapest half-hour
   slots of each day up to the requested charge-window duration and power limit.
4. Standing charges are included when provided.
5. This is an economic model, not a full physical battery simulator.

References
----------
- Octopus public REST API base and product endpoints:
  https://developer.octopus.energy/rest/guides/api-basics
  https://developer.octopus.energy/rest/guides/endpoints
- Tesla Fleet API energy endpoints (official):
  https://developer.tesla.com/docs/fleet-api/endpoints/energy
- TeslaPy project (community / unofficial owner API helper):
  https://github.com/tdorssers/TeslaPy
"""

from __future__ import annotations

import argparse
import csv
import getpass
import json
import math
import os
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    from backports.zoneinfo import ZoneInfo  # type: ignore


LOCAL_TZ = ZoneInfo("Europe/London")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class IntelligentTariff:
    offpeak_import_p_per_kwh: float = 7.0
    peak_import_p_per_kwh: float = 26.0
    export_p_per_kwh: float = 15.0
    standing_charge_p_per_day: float = 0.0
    offpeak_start: str = "23:30"
    offpeak_end: str = "05:30"


@dataclass
class AgileConfig:
    standing_charge_p_per_day: float = 0.0
    flexible_charge_hours_per_day: float = 6.0
    flexible_max_kw: float = 3.3


@dataclass
class ScenarioConfig:
    battery_capacity_kwh: float = 13.0
    extra_daily_kwh: float = 0.0
    extra_start: str = "17:00"
    extra_end: str = "22:00"
    timezone: str = "Europe/London"


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def parse_hhmm(value: str) -> Tuple[int, int]:
    try:
        hh, mm = value.split(":")
        hh_i = int(hh)
        mm_i = int(mm)
    except Exception as exc:
        raise ValueError(f"Invalid time '{value}', expected HH:MM") from exc
    if not (0 <= hh_i <= 23 and 0 <= mm_i <= 59):
        raise ValueError(f"Invalid time '{value}', expected HH:MM")
    return hh_i, mm_i


def time_in_window(ts: pd.Timestamp, start_hhmm: str, end_hhmm: str) -> bool:
    sh, sm = parse_hhmm(start_hhmm)
    eh, em = parse_hhmm(end_hhmm)
    tmins = ts.hour * 60 + ts.minute
    smins = sh * 60 + sm
    emins = eh * 60 + em
    if smins < emins:
        return smins <= tmins < emins
    return tmins >= smins or tmins < emins


def ensure_localized(ts: pd.Series, timezone: str = "Europe/London") -> pd.Series:
    """Assume naive Tesla timestamps are local wall clock time."""
    tz = ZoneInfo(timezone)
    if getattr(ts.dt, "tz", None) is None:
        return ts.dt.tz_localize(
            tz,
            ambiguous="infer",
            nonexistent="shift_forward",
        )
    return ts.dt.tz_convert(tz)


def half_hour_floor(ts: pd.Series) -> pd.Series:
    return ts.dt.floor("30min")


def pence_to_pounds(pence: float) -> float:
    return pence / 100.0


# ---------------------------------------------------------------------------
# Tariff file discovery / loading
# ---------------------------------------------------------------------------

TARIFF_RE = re.compile(
    r"^csv_(?P<kind>agile|agileoutgoing)_(?P<region_code>[A-Z])_(?P<region_name>.+)\.csv$",
    re.IGNORECASE,
)


def discover_tariff_files(folder: Path) -> List[dict]:
    records: List[dict] = []
    for path in sorted(folder.glob("csv_*_*.csv")):
        match = TARIFF_RE.match(path.name)
        if not match:
            continue
        rec = match.groupdict()
        rec["path"] = str(path)
        records.append(rec)
    return records


def read_agile_csv(path: Path, timezone: str = "Europe/London") -> pd.DataFrame:
    """
    Read an Agile CSV saved in the common 5-column format:
    timestamp_utc, local_time, region_code, region_name, price_p_per_kwh

    Some files arrive without a header row, so this loader detects that.
    """
    raw = pd.read_csv(path, header=None)
    if raw.shape[1] < 5:
        raise ValueError(f"{path} does not look like a 5-column Agile CSV")

    # If the first row is a header row, drop it
    first_val = str(raw.iloc[0, 0]).strip().lower()
    if first_val in {"timestamp", "period_start", "valid_from", "datetime"}:
        raw = raw.iloc[1:].copy()

    raw = raw.iloc[:, :5].copy()
    raw.columns = ["timestamp", "local_time", "region_code", "region_name", "price_p_per_kwh"]

    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
    raw["price_p_per_kwh"] = pd.to_numeric(raw["price_p_per_kwh"], errors="coerce")
    raw = raw.dropna(subset=["timestamp", "price_p_per_kwh"]).copy()

    tz = ZoneInfo(timezone)
    raw["timestamp_local"] = raw["timestamp"].dt.tz_convert(tz)
    raw["slot_start"] = raw["timestamp_local"].dt.floor("30min")
    raw = raw.drop_duplicates(subset=["slot_start"], keep="last")
    return raw[["slot_start", "price_p_per_kwh", "region_code", "region_name"]].sort_values("slot_start")


# ---------------------------------------------------------------------------
# Tesla CSV loading
# ---------------------------------------------------------------------------

def load_power_csv(path: Path, timezone: str = "Europe/London") -> pd.DataFrame:
    """
    Load Tesla 5-minute power export CSV.

    Expected columns:
    - timestamp
    - grid_power
    Optional:
    - solar_power
    - battery_power
    - load_power
    """
    df = pd.read_csv(path)
    if "timestamp" not in df.columns or "grid_power" not in df.columns:
        raise ValueError(f"{path} must contain 'timestamp' and 'grid_power' columns")

    # Clean embedded header rows / malformed values
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["grid_power"] = pd.to_numeric(df["grid_power"], errors="coerce")
    for col in ("solar_power", "battery_power", "load_power"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["timestamp", "grid_power"]).copy()
    df["timestamp_local"] = ensure_localized(df["timestamp"], timezone)

    # 5-minute power to energy
    df["import_kwh_5m"] = np.where(df["grid_power"] > 0, df["grid_power"] / 12.0 / 1000.0, 0.0)
    df["export_kwh_5m"] = np.where(df["grid_power"] < 0, -df["grid_power"] / 12.0 / 1000.0, 0.0)

    if "load_power" in df.columns:
        df["load_kwh_5m"] = np.where(df["load_power"] > 0, df["load_power"] / 12.0 / 1000.0, 0.0)
    else:
        df["load_kwh_5m"] = np.nan

    df["slot_start"] = half_hour_floor(df["timestamp_local"])

    hh = (
        df.groupby("slot_start", as_index=False)
        .agg(
            import_kwh=("import_kwh_5m", "sum"),
            export_kwh=("export_kwh_5m", "sum"),
            load_kwh=("load_kwh_5m", "sum"),
        )
        .sort_values("slot_start")
    )
    hh["date"] = hh["slot_start"].dt.date
    return hh


def trim_date_range(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    out = df.copy()
    if start:
        start_ts = pd.Timestamp(start, tz=LOCAL_TZ)
        out = out[out["slot_start"] >= start_ts]
    if end:
        end_ts = pd.Timestamp(end, tz=LOCAL_TZ) + pd.Timedelta(days=1)
        out = out[out["slot_start"] < end_ts]
    return out


# ---------------------------------------------------------------------------
# Optional Tesla fetchers
# ---------------------------------------------------------------------------

def fetch_tesla_power_history_via_teslapy(
    email: str,
    site_id: Optional[str],
    start: str,
    end: str,
    out_csv: Path,
) -> Path:
    """
    Best-effort fetch via TeslaPy.

    TeslaPy is a community library built on Tesla's owner-facing APIs, and
    the exact energy-history responses may change. This helper intentionally
    keeps the fetcher lightweight: if the account / endpoint shape differs,
    users can still fall back to CSV export mode.
    """
    try:
        import teslapy  # type: ignore
    except ImportError as exc:
        raise RuntimeError("TeslaPy is not installed. `pip install teslapy` first.") from exc

    with teslapy.Tesla(email) as tesla:
        products = tesla.battery_list()
        if not products:
            raise RuntimeError("No Tesla energy products found for this account.")

        battery = None
        if site_id:
            for product in products:
                sid = str(product.get("energy_site_id") or product.get("site_id") or "")
                if sid == str(site_id):
                    battery = product
                    break
        if battery is None:
            battery = products[0]

        # Most community examples use battery.api() for raw owner API calls.
        # The route shape can vary over time, so this is intentionally wrapped
        # in a helpful error if unavailable.
        try:
            energy_site_id = battery["energy_site_id"]
            payload = battery.api(
                f"api/1/energy_sites/{energy_site_id}/history",
                kind="power",
                period="day",
                start_date=start,
                end_date=end,
            )
        except Exception as exc:
            raise RuntimeError(
                "TeslaPy fetch failed. The account may require a different "
                "auth flow or the owner API response shape may have changed. "
                "Use CSV export mode instead."
            ) from exc

        # Attempt to normalise likely response shapes
        time_series = (
            payload.get("response", {}).get("time_series")
            or payload.get("time_series")
            or payload.get("series")
            or []
        )
        if not time_series:
            raise RuntimeError("TeslaPy returned no power time-series data.")

        rows = []
        for item in time_series:
            rows.append(
                {
                    "timestamp": item.get("timestamp") or item.get("time") or item.get("instant"),
                    "solar_power": item.get("solar_power"),
                    "battery_power": item.get("battery_power"),
                    "grid_power": item.get("grid_power") or item.get("site_power"),
                    "load_power": item.get("load_power"),
                }
            )
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv


def fetch_powerwall_via_local_gateway(
    host: str,
    password: str,
    out_csv: Path,
    samples: int = 288,
    poll_seconds: int = 300,
) -> Path:
    """
    Fetch live local-gateway samples via pyPowerwall.

    This is for creating a fresh power CSV over time, not backfilling history.
    """
    try:
        import pypowerwall  # type: ignore
    except ImportError as exc:
        raise RuntimeError("pyPowerwall is not installed. `pip install pypowerwall` first.") from exc

    import time

    pw = pypowerwall.Powerwall(host, password=password)
    rows = []
    for _ in range(samples):
        status = pw.power()
        rows.append(
            {
                "timestamp": pd.Timestamp.now(tz=LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S"),
                "solar_power": status.get("solar"),
                "battery_power": status.get("battery"),
                "grid_power": status.get("site"),
                "load_power": status.get("load"),
            }
        )
        time.sleep(poll_seconds)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv


# ---------------------------------------------------------------------------
# Extra usage profile
# ---------------------------------------------------------------------------

def build_extra_profile(
    slots: pd.Series,
    extra_daily_kwh: float,
    start_hhmm: str,
    end_hhmm: str,
) -> pd.Series:
    """
    Spread extra kWh/day evenly across half-hour slots within the selected
    local-time window.
    """
    profile = pd.Series(0.0, index=slots.index)
    if extra_daily_kwh <= 0:
        return profile

    slot_df = pd.DataFrame({"slot_start": slots})
    slot_df["date"] = slot_df["slot_start"].dt.date
    slot_df["in_window"] = slot_df["slot_start"].apply(lambda ts: time_in_window(ts, start_hhmm, end_hhmm))

    for _, day_idx in slot_df.groupby("date").groups.items():
        day_slots = slot_df.loc[list(day_idx)]
        eligible_idx = day_slots.index[day_slots["in_window"]].tolist()
        if not eligible_idx:
            continue
        per_slot = extra_daily_kwh / len(eligible_idx)
        profile.loc[eligible_idx] = per_slot
    return profile


# ---------------------------------------------------------------------------
# Intelligent model
# ---------------------------------------------------------------------------

def calculate_intelligent(
    hh: pd.DataFrame,
    tariff: IntelligentTariff,
    scenario: ScenarioConfig,
) -> Tuple[dict, pd.DataFrame]:
    """
    Model Intelligent with a hard daily cheap-energy cap equal to battery capacity.

    Method:
    - Existing direct grid import remains as historically observed.
    - Add any extra usage into the selected time window.
    - For each day:
      * cheap battery-served energy available = battery_capacity_kwh
      * direct import observed during offpeak window is priced at offpeak
      * direct import observed outside offpeak is priced at peak
      * extra usage can consume cheap battery energy first; any remainder is peak
    This preserves the user's requested 'once the battery is empty, it is 26p'.
    """
    df = hh.copy()
    df["date"] = df["slot_start"].dt.date
    df["extra_kwh"] = build_extra_profile(
        df["slot_start"],
        scenario.extra_daily_kwh,
        scenario.extra_start,
        scenario.extra_end,
    )
    df["is_offpeak"] = df["slot_start"].apply(lambda ts: time_in_window(ts, tariff.offpeak_start, tariff.offpeak_end))

    daily_rows = []
    total_import_cost_p = 0.0
    total_export_revenue_p = 0.0
    total_sc_p = 0.0

    for day, group in df.groupby("date", sort=True):
        group = group.copy().sort_values("slot_start")

        existing_import_offpeak = group.loc[group["is_offpeak"], "import_kwh"].sum()
        existing_import_peak = group.loc[~group["is_offpeak"], "import_kwh"].sum()
        export_kwh = group["export_kwh"].sum()
        extra_kwh = group["extra_kwh"].sum()

        # Cheap stored energy available once per day
        cheap_battery_kwh_available = scenario.battery_capacity_kwh

        # We treat extra usage as being served from available cheap battery energy first.
        # Any spill is charged at peak rate.
        extra_cheap_kwh = min(extra_kwh, cheap_battery_kwh_available)
        extra_expensive_kwh = max(extra_kwh - cheap_battery_kwh_available, 0.0)

        # Historical direct import is costed as observed
        import_cost_p = (
            existing_import_offpeak * tariff.offpeak_import_p_per_kwh
            + existing_import_peak * tariff.peak_import_p_per_kwh
            + extra_cheap_kwh * tariff.offpeak_import_p_per_kwh
            + extra_expensive_kwh * tariff.peak_import_p_per_kwh
        )
        export_revenue_p = export_kwh * tariff.export_p_per_kwh
        sc_p = tariff.standing_charge_p_per_day

        total_import_cost_p += import_cost_p
        total_export_revenue_p += export_revenue_p
        total_sc_p += sc_p

        daily_rows.append(
            {
                "date": day,
                "existing_import_offpeak_kwh": round(existing_import_offpeak, 6),
                "existing_import_peak_kwh": round(existing_import_peak, 6),
                "export_kwh": round(export_kwh, 6),
                "extra_kwh": round(extra_kwh, 6),
                "cheap_battery_kwh_available": round(cheap_battery_kwh_available, 6),
                "extra_cheap_kwh": round(extra_cheap_kwh, 6),
                "extra_expensive_kwh": round(extra_expensive_kwh, 6),
                "battery_ran_out": extra_expensive_kwh > 0,
                "import_cost_p": round(import_cost_p, 4),
                "export_revenue_p": round(export_revenue_p, 4),
                "standing_charge_p": round(sc_p, 4),
                "net_cost_p": round(import_cost_p + sc_p - export_revenue_p, 4),
            }
        )

    summary = {
        "tariff": "intelligent",
        "days": len(daily_rows),
        "import_cost_gbp": round(pence_to_pounds(total_import_cost_p), 2),
        "export_revenue_gbp": round(pence_to_pounds(total_export_revenue_p), 2),
        "standing_charge_gbp": round(pence_to_pounds(total_sc_p), 2),
        "net_cost_gbp": round(pence_to_pounds(total_import_cost_p + total_sc_p - total_export_revenue_p), 2),
    }
    return summary, pd.DataFrame(daily_rows)


# ---------------------------------------------------------------------------
# Agile model
# ---------------------------------------------------------------------------

def _allocate_flexible_energy_to_cheapest_slots(
    day_df: pd.DataFrame,
    flexible_kwh: float,
    max_kw: float,
) -> pd.Series:
    """
    Allocate flexible energy into the cheapest slots for the day, capped by
    max_kw per slot.
    """
    alloc = pd.Series(0.0, index=day_df.index)
    if flexible_kwh <= 0:
        return alloc

    # Each half-hour slot capacity in kWh
    slot_cap_kwh = max_kw * 0.5

    sorted_idx = day_df.sort_values("agile_import_p_per_kwh").index.tolist()
    remaining = flexible_kwh

    for idx in sorted_idx:
        if remaining <= 1e-9:
            break
        add = min(slot_cap_kwh, remaining)
        alloc.loc[idx] += add
        remaining -= add

    return alloc


def calculate_agile(
    hh: pd.DataFrame,
    agile_import: pd.DataFrame,
    agile_export: pd.DataFrame,
    config: AgileConfig,
    scenario: ScenarioConfig,
) -> Tuple[dict, pd.DataFrame]:
    """
    Agile smart-shift model:
    - Historical direct import/export is costed against actual half-hour prices.
    - Flexible demand = extra usage only by default (manual adjustable load).
    - Additional smart shifting can also include a user-specified amount of
      flexible battery/grid charging if they choose to represent it inside the
      historical import stream externally.

    For simplicity and auditability, this script treats the *manual extra usage*
    as flexible by default and moves it into the cheapest slots of each day.
    """
    df = hh.copy()
    df = df.merge(
        agile_import[["slot_start", "price_p_per_kwh"]].rename(columns={"price_p_per_kwh": "agile_import_p_per_kwh"}),
        on="slot_start",
        how="left",
    ).merge(
        agile_export[["slot_start", "price_p_per_kwh"]].rename(columns={"price_p_per_kwh": "agile_export_p_per_kwh"}),
        on="slot_start",
        how="left",
    )
    if df["agile_import_p_per_kwh"].isna().any():
        missing = int(df["agile_import_p_per_kwh"].isna().sum())
        raise ValueError(f"Missing Agile import price for {missing} half-hour slots")
    if df["agile_export_p_per_kwh"].isna().any():
        missing = int(df["agile_export_p_per_kwh"].isna().sum())
        raise ValueError(f"Missing Agile export price for {missing} half-hour slots")

    df["date"] = df["slot_start"].dt.date
    df["extra_kwh"] = build_extra_profile(
        df["slot_start"],
        scenario.extra_daily_kwh,
        scenario.extra_start,
        scenario.extra_end,
    )

    daily_rows = []
    total_import_cost_p = 0.0
    total_export_revenue_p = 0.0
    total_sc_p = 0.0

    # Maximum flexible energy per day based on chosen charge window and power limit
    flexible_daily_cap_kwh = config.flexible_charge_hours_per_day * config.flexible_max_kw

    for day, group in df.groupby("date", sort=True):
        group = group.copy().sort_values("slot_start")
        direct_import_kwh = group["import_kwh"].sum()
        export_kwh = group["export_kwh"].sum()
        extra_kwh = group["extra_kwh"].sum()

        flexible_kwh = min(extra_kwh, flexible_daily_cap_kwh)
        stranded_expensive_kwh = max(extra_kwh - flexible_daily_cap_kwh, 0.0)

        shifted = _allocate_flexible_energy_to_cheapest_slots(group, flexible_kwh, config.flexible_max_kw)
        direct_cost_p = float((group["import_kwh"] * group["agile_import_p_per_kwh"]).sum())
        shifted_cost_p = float((shifted * group["agile_import_p_per_kwh"]).sum())
        stranded_cost_p = 0.0

        # Anything beyond flexible capacity stays in the user's requested window
        # and is costed in those original slots.
        if stranded_expensive_kwh > 1e-9:
            remaining = stranded_expensive_kwh
            window_idx = group.index[group["extra_kwh"] > 0].tolist()
            for idx in window_idx:
                if remaining <= 1e-9:
                    break
                base = group.loc[idx, "extra_kwh"]
                use = min(base, remaining)
                stranded_cost_p += use * group.loc[idx, "agile_import_p_per_kwh"]
                remaining -= use

        export_revenue_p = float((group["export_kwh"] * group["agile_export_p_per_kwh"]).sum())
        sc_p = config.standing_charge_p_per_day

        import_cost_p = direct_cost_p + shifted_cost_p + stranded_cost_p

        total_import_cost_p += import_cost_p
        total_export_revenue_p += export_revenue_p
        total_sc_p += sc_p

        daily_rows.append(
            {
                "date": day,
                "direct_import_kwh": round(direct_import_kwh, 6),
                "export_kwh": round(export_kwh, 6),
                "extra_kwh": round(extra_kwh, 6),
                "flexible_shifted_kwh": round(flexible_kwh, 6),
                "stranded_in_original_window_kwh": round(stranded_expensive_kwh, 6),
                "battery_ran_out": stranded_expensive_kwh > 0,
                "avg_agile_import_p_per_kwh": round(float(group["agile_import_p_per_kwh"].mean()), 4),
                "avg_agile_export_p_per_kwh": round(float(group["agile_export_p_per_kwh"].mean()), 4),
                "import_cost_p": round(import_cost_p, 4),
                "export_revenue_p": round(export_revenue_p, 4),
                "standing_charge_p": round(sc_p, 4),
                "net_cost_p": round(import_cost_p + sc_p - export_revenue_p, 4),
            }
        )

    summary = {
        "tariff": "agile",
        "days": len(daily_rows),
        "import_cost_gbp": round(pence_to_pounds(total_import_cost_p), 2),
        "export_revenue_gbp": round(pence_to_pounds(total_export_revenue_p), 2),
        "standing_charge_gbp": round(pence_to_pounds(total_sc_p), 2),
        "net_cost_gbp": round(pence_to_pounds(total_import_cost_p + total_sc_p - total_export_revenue_p), 2),
    }
    return summary, pd.DataFrame(daily_rows)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def print_summary_table(summaries: List[dict]) -> None:
    if not summaries:
        return
    headers = ["tariff", "import_cost_gbp", "export_revenue_gbp", "standing_charge_gbp", "net_cost_gbp"]
    widths = {h: max(len(h), max(len(str(s.get(h, ""))) for s in summaries)) for h in headers}
    line = "  ".join(h.ljust(widths[h]) for h in headers)
    print(line)
    print("  ".join("-" * widths[h] for h in headers))
    for s in summaries:
        print("  ".join(str(s.get(h, "")).ljust(widths[h]) for h in headers))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare Tesla Powerwall historical usage against Octopus tariffs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    sub = parser.add_subparsers(dest="command", required=True)

    discover = sub.add_parser("discover-tariffs", help="List Agile tariff CSV files found in a folder")
    discover.add_argument("--tariff-dir", required=True, type=Path)

    compare = sub.add_parser("compare", help="Run the tariff comparison")
    compare.add_argument("--power-csv", type=Path, help="Tesla 5-minute power CSV export")
    compare.add_argument("--tariff-dir", required=True, type=Path, help="Folder containing saved Agile CSVs")
    compare.add_argument("--region-code", default="M", help="Octopus region code, e.g. M")
    compare.add_argument("--region-name", default="Yorkshire", help="Region name used in saved CSV filenames")
    compare.add_argument("--start-date", help="Start date YYYY-MM-DD")
    compare.add_argument("--end-date", help="End date YYYY-MM-DD")

    compare.add_argument("--battery-capacity-kwh", type=float, default=13.0)
    compare.add_argument("--extra-daily-kwh", type=float, default=0.0)
    compare.add_argument("--extra-start", default="17:00")
    compare.add_argument("--extra-end", default="22:00")

    compare.add_argument("--intelligent-offpeak-import", type=float, default=7.0)
    compare.add_argument("--intelligent-peak-import", type=float, default=26.0)
    compare.add_argument("--intelligent-export", type=float, default=15.0)
    compare.add_argument("--intelligent-standing-charge", type=float, default=0.0)
    compare.add_argument("--intelligent-offpeak-start", default="23:30")
    compare.add_argument("--intelligent-offpeak-end", default="05:30")

    compare.add_argument("--agile-standing-charge", type=float, default=0.0)
    compare.add_argument("--agile-flex-hours", type=float, default=6.0)
    compare.add_argument("--agile-flex-max-kw", type=float, default=3.3)

    compare.add_argument("--out-dir", type=Path, default=Path("output"))
    compare.add_argument("--write-half-hour-audit", action="store_true")

    fetch = sub.add_parser("fetch-tesla", help="Fetch Tesla data to CSV using TeslaPy or pyPowerwall")
    fetch.add_argument("--mode", choices=["teslapy", "local-gateway"], required=True)
    fetch.add_argument("--email", help="Tesla account email for TeslaPy")
    fetch.add_argument("--site-id", help="Tesla energy site ID for TeslaPy")
    fetch.add_argument("--start-date", help="Start date YYYY-MM-DD for TeslaPy")
    fetch.add_argument("--end-date", help="End date YYYY-MM-DD for TeslaPy")
    fetch.add_argument("--gateway-host", help="Local Powerwall gateway host/IP")
    fetch.add_argument("--gateway-password", help="Local Powerwall installer/customer password")
    fetch.add_argument("--samples", type=int, default=288, help="Samples to capture in local-gateway mode")
    fetch.add_argument("--poll-seconds", type=int, default=300)
    fetch.add_argument("--out-csv", type=Path, required=True)

    return parser


def discover_paths(tariff_dir: Path, region_code: str, region_name: str) -> Tuple[Path, Path]:
    region_code = region_code.upper()
    in_path = tariff_dir / f"csv_agile_{region_code}_{region_name}.csv"
    out_path = tariff_dir / f"csv_agileoutgoing_{region_code}_{region_name}.csv"
    if not in_path.exists():
        raise FileNotFoundError(f"Agile import CSV not found: {in_path}")
    if not out_path.exists():
        raise FileNotFoundError(f"Agile outgoing CSV not found: {out_path}")
    return in_path, out_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "discover-tariffs":
        records = discover_tariff_files(args.tariff_dir)
        if not records:
            print("No matching tariff CSVs found.")
            return 1
        print(json.dumps(records, indent=2))
        return 0

    if args.command == "fetch-tesla":
        if args.mode == "teslapy":
            if not args.email or not args.start_date or not args.end_date:
                parser.error("--email, --start-date and --end-date are required for teslapy mode")
            try:
                path = fetch_tesla_power_history_via_teslapy(
                    email=args.email,
                    site_id=args.site_id,
                    start=args.start_date,
                    end=args.end_date,
                    out_csv=args.out_csv,
                )
            except Exception as exc:
                print(f"TeslaPy fetch failed: {exc}", file=sys.stderr)
                return 2
            print(f"Wrote {path}")
            return 0

        if args.mode == "local-gateway":
            if not args.gateway_host:
                parser.error("--gateway-host is required for local-gateway mode")
            password = args.gateway_password or getpass.getpass("Powerwall gateway password: ")
            try:
                path = fetch_powerwall_via_local_gateway(
                    host=args.gateway_host,
                    password=password,
                    out_csv=args.out_csv,
                    samples=args.samples,
                    poll_seconds=args.poll_seconds,
                )
            except Exception as exc:
                print(f"Local gateway fetch failed: {exc}", file=sys.stderr)
                return 2
            print(f"Wrote {path}")
            return 0

    if args.command == "compare":
        if not args.power_csv:
            parser.error("--power-csv is required for compare")

        hh = load_power_csv(args.power_csv)
        hh = trim_date_range(hh, args.start_date, args.end_date)
        if hh.empty:
            raise RuntimeError("No power data left after applying the date range")

        in_path, out_path = discover_paths(args.tariff_dir, args.region_code, args.region_name)
        agile_import = read_agile_csv(in_path)
        agile_export = read_agile_csv(out_path)

        scenario = ScenarioConfig(
            battery_capacity_kwh=args.battery_capacity_kwh,
            extra_daily_kwh=args.extra_daily_kwh,
            extra_start=args.extra_start,
            extra_end=args.extra_end,
        )

        intelligent_tariff = IntelligentTariff(
            offpeak_import_p_per_kwh=args.intelligent_offpeak_import,
            peak_import_p_per_kwh=args.intelligent_peak_import,
            export_p_per_kwh=args.intelligent_export,
            standing_charge_p_per_day=args.intelligent_standing_charge,
            offpeak_start=args.intelligent_offpeak_start,
            offpeak_end=args.intelligent_offpeak_end,
        )
        agile_cfg = AgileConfig(
            standing_charge_p_per_day=args.agile_standing_charge,
            flexible_charge_hours_per_day=args.agile_flex_hours,
            flexible_max_kw=args.agile_flex_max_kw,
        )

        intelligent_summary, intelligent_daily = calculate_intelligent(hh, intelligent_tariff, scenario)
        agile_summary, agile_daily = calculate_agile(hh, agile_import, agile_export, agile_cfg, scenario)

        args.out_dir.mkdir(parents=True, exist_ok=True)
        write_csv(intelligent_daily, args.out_dir / "daily_breakdown_intelligent.csv")
        write_csv(agile_daily, args.out_dir / "daily_breakdown_agile.csv")
        if args.write_half_hour_audit:
            write_csv(hh, args.out_dir / "half_hour_input.csv")

        summary = pd.DataFrame([intelligent_summary, agile_summary]).sort_values("net_cost_gbp")
        write_csv(summary, args.out_dir / "summary.csv")
        print_summary_table(summary.to_dict(orient="records"))
        print(f"\nBest tariff by model: {summary.iloc[0]['tariff']}")
        print(f"Outputs written to: {args.out_dir}")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
