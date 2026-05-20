#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore

LOCAL_TZ = ZoneInfo("Europe/London")
DEFAULT_TARIFF_INDEX = "https://files.energy-stats.uk/csv_output/"

DEFAULTS_FILE = Path(".octopus_defaults.json")

# Keys that can be saved as user defaults
SAVEABLE_DEFAULTS = {
    "region_code", "battery_capacity_kwh", "extra_daily_kwh",
    "extra_start", "extra_end", "email", "out_dir", "cache_dir",
    "intelligent_offpeak_import", "intelligent_peak_import",
    "intelligent_export", "intelligent_standing_charge",
    "intelligent_offpeak_start", "intelligent_offpeak_end",
    "agile_standing_charge", "agile_flex_hours", "agile_flex_max_kw",
    "ev_exclusion_enabled", "ev_min_power_w", "ev_start", "ev_end",
    "go_offpeak_import", "go_peak_import", "go_export", "go_standing_charge",
    "go_offpeak_start", "go_offpeak_end",
    "flux_offpeak_import", "flux_day_import", "flux_peak_import",
    "flux_offpeak_export", "flux_day_export", "flux_peak_export",
    "flux_standing_charge", "flux_offpeak_start", "flux_offpeak_end",
    "flux_peak_start", "flux_peak_end",
    "cosy_import", "cosy_day_import", "cosy_peak_import",
    "cosy_export", "cosy_standing_charge",
    "flexible_import", "flexible_export", "flexible_standing_charge",
    "tracker_standing_charge", "tracker_export",
    "tariffs",
}


def load_defaults() -> dict:
    """Load saved defaults from .octopus_defaults.json if it exists."""
    if DEFAULTS_FILE.exists():
        with open(DEFAULTS_FILE) as f:
            return json.load(f)
    return {}


def save_defaults(defaults: dict) -> None:
    """Save defaults to .octopus_defaults.json."""
    with open(DEFAULTS_FILE, "w") as f:
        json.dump(defaults, f, indent=2)


def apply_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Apply saved defaults to any argument that wasn't explicitly set on the command line."""
    saved = load_defaults()
    for key, value in saved.items():
        if key in SAVEABLE_DEFAULTS and getattr(args, key, None) is None:
            setattr(args, key, value)
    return args

REGION_CODE_TO_NAME = {
    "A": "Eastern_England",
    "B": "East_Midlands",
    "C": "London",
    "D": "Merseyside_and_Northern_Wales",
    "E": "West_Midlands",
    "F": "North_Eastern_England",
    "G": "North_Western_England",
    "H": "Southern_England",
    "J": "South_Eastern_England",
    "K": "Southern_Wales",
    "L": "South_Western_England",
    "M": "Yorkshire",
    "N": "Southern_Scotland",
    "P": "Northern_Scotland",
}

INDEX_LINK_RE = re.compile(r'href=\"([^\"]+\.csv)\"', re.IGNORECASE)


@dataclass
class IntelligentTariff:
    offpeak_import_p_per_kwh: float = 7.0
    peak_import_p_per_kwh: float = 26.0
    export_p_per_kwh: float = 15.0
    standing_charge_p_per_day: float = 57.01
    offpeak_start: str = "23:30"
    offpeak_end: str = "05:30"


@dataclass
class GoTariff:
    """Octopus Go — cheap overnight rate for EV owners."""
    offpeak_import_p_per_kwh: float = 8.0
    peak_import_p_per_kwh: float = 24.5
    export_p_per_kwh: float = 12.0
    standing_charge_p_per_day: float = 53.35
    offpeak_start: str = "23:30"
    offpeak_end: str = "05:30"


@dataclass
class FluxTariff:
    """Octopus Flux — 3 time bands for import and export, designed for solar+battery."""
    offpeak_import_p_per_kwh: float = 9.80
    day_import_p_per_kwh: float = 22.36
    peak_import_p_per_kwh: float = 33.54
    offpeak_export_p_per_kwh: float = 4.05
    day_export_p_per_kwh: float = 14.40
    peak_export_p_per_kwh: float = 28.60
    standing_charge_p_per_day: float = 48.93
    offpeak_start: str = "02:00"
    offpeak_end: str = "05:00"
    peak_start: str = "16:00"
    peak_end: str = "19:00"


@dataclass
class CosyTariff:
    """Octopus Cosy — 3 cheap windows for heat pumps, peak 16:00-19:00."""
    cosy_import_p_per_kwh: float = 12.0
    day_import_p_per_kwh: float = 24.50
    peak_import_p_per_kwh: float = 36.75
    export_p_per_kwh: float = 12.0
    standing_charge_p_per_day: float = 53.35
    # Cosy windows (cheap rate)
    cosy_window_1_start: str = "04:00"
    cosy_window_1_end: str = "07:00"
    cosy_window_2_start: str = "13:00"
    cosy_window_2_end: str = "16:00"
    cosy_window_3_start: str = "22:00"
    cosy_window_3_end: str = "00:00"
    # Peak window (expensive rate)
    peak_start: str = "16:00"
    peak_end: str = "19:00"


@dataclass
class FlexibleTariff:
    """Octopus Flexible (SVR) — flat rate standard variable tariff."""
    import_p_per_kwh: float = 24.50
    export_p_per_kwh: float = 12.0
    standing_charge_p_per_day: float = 53.35


@dataclass
class TrackerTariff:
    """Octopus Tracker — daily variable rate tied to wholesale prices."""
    standing_charge_p_per_day: float = 53.35
    export_p_per_kwh: float = 12.0
    # Tracker uses daily rates from CSV/API — no fixed import rate


@dataclass
class AgileConfig:
    standing_charge_p_per_day: float = 66.26
    flexible_charge_hours_per_day: float = 6.0
    flexible_max_kw: float = 3.3


@dataclass
class ScenarioConfig:
    battery_capacity_kwh: float = 13.0
    extra_daily_kwh: float = 0.0
    extra_start: str = "17:00"
    extra_end: str = "22:00"
    timezone: str = "Europe/London"
    ev_exclusion_enabled: bool = True
    ev_min_power_w: float = 5500.0
    ev_start: str = "23:30"
    ev_end: str = "04:30"


def parse_hhmm(value: str) -> Tuple[int, int]:
    hh, mm = value.split(":")
    hh_i, mm_i = int(hh), int(mm)
    if not (0 <= hh_i <= 23 and 0 <= mm_i <= 59):
        raise ValueError(f"Invalid time {value}")
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
    tz = ZoneInfo(timezone)
    if getattr(ts.dt, "tz", None) is None:
        return ts.dt.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
    return ts.dt.tz_convert(tz)


def pence_to_pounds(value: float) -> float:
    return value / 100.0


def fetch_tariff_index(index_url: str = DEFAULT_TARIFF_INDEX) -> List[str]:
    response = requests.get(index_url, timeout=30)
    response.raise_for_status()
    return sorted(set(INDEX_LINK_RE.findall(response.text)))


def region_name_from_code(region_code: str) -> str:
    code = region_code.upper()
    if code not in REGION_CODE_TO_NAME:
        raise ValueError(f"Unknown region code '{region_code}'")
    return REGION_CODE_TO_NAME[code]


def download_region_tariffs(
    region_code: str, cache_dir: Path,
    index_url: str = DEFAULT_TARIFF_INDEX, force: bool = False,
) -> Tuple[Path, Path]:
    region_code = region_code.upper()
    region_name = region_name_from_code(region_code)
    import_name = f"csv_agile_{region_code}_{region_name}.csv"
    export_name = f"csv_agileoutgoing_{region_code}_{region_name}.csv"

    available = fetch_tariff_index(index_url)
    missing = [name for name in (import_name, export_name) if name not in available]
    if missing:
        raise FileNotFoundError(f"Could not find tariff CSV(s): {missing}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename in (import_name, export_name):
        dest = cache_dir / filename
        if force or not dest.exists():
            resp = requests.get(urljoin(index_url, filename), timeout=60)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
        paths.append(dest)
    return paths[0], paths[1]


def read_agile_csv(path: Path, timezone: str = "Europe/London") -> pd.DataFrame:
    raw = pd.read_csv(path, header=None)
    if raw.shape[1] < 5:
        raise ValueError(f"{path} does not look like a 5-column Agile CSV")
    first_val = str(raw.iloc[0, 0]).strip().lower()
    if first_val in {"timestamp", "period_start", "valid_from", "datetime"}:
        raw = raw.iloc[1:].copy()
    raw = raw.iloc[:, :5].copy()
    raw.columns = ["timestamp", "local_time", "region_code", "region_name", "price_p_per_kwh"]
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
    raw["price_p_per_kwh"] = pd.to_numeric(raw["price_p_per_kwh"], errors="coerce")
    raw = raw.dropna(subset=["timestamp", "price_p_per_kwh"]).copy()
    tz = ZoneInfo(timezone)
    # Handle DST transitions by inferring ambiguous times and shifting forward for nonexistent times
    raw["slot_start"] = (
        raw["timestamp"]
        .dt.tz_convert(tz)
        .dt.floor("30min", ambiguous="infer", nonexistent="shift_forward")
    )
    raw = raw.drop_duplicates(subset=["slot_start"], keep="last")
    return raw[["slot_start", "price_p_per_kwh", "region_code", "region_name"]].sort_values("slot_start")


def load_power_csv(path: Path, scenario: ScenarioConfig) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns or "grid_power" not in df.columns:
        raise ValueError("Power CSV must include timestamp and grid_power columns")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["grid_power"] = pd.to_numeric(df["grid_power"], errors="coerce")
    df = df.dropna(subset=["timestamp", "grid_power"]).copy()
    tz = ZoneInfo(scenario.timezone)
    # Floor in UTC first (no DST ambiguity), then convert to local
    df["timestamp_local"] = df["timestamp"].dt.tz_convert(tz)
    df["slot_start"] = df["timestamp"].dt.floor("30min").dt.tz_convert(tz)

    df["import_kwh_5m"] = np.where(df["grid_power"] > 0, df["grid_power"] / 12.0 / 1000.0, 0.0)
    df["export_kwh_5m"] = np.where(df["grid_power"] < 0, -df["grid_power"] / 12.0 / 1000.0, 0.0)

    # Track battery charging from grid: battery_power < 0 means charging
    # The portion from grid is min(|battery_power|, grid_power) when both conditions hold
    if "battery_power" in df.columns:
        df["battery_power"] = pd.to_numeric(df["battery_power"], errors="coerce").fillna(0.0)
        df["battery_charge_from_grid_kwh_5m"] = np.where(
            (df["battery_power"] < 0) & (df["grid_power"] > 0),
            np.minimum(np.abs(df["battery_power"]), df["grid_power"]) / 12.0 / 1000.0,
            0.0,
        )
    else:
        df["battery_charge_from_grid_kwh_5m"] = 0.0

    # Track solar generation and home load for full simulation
    if "solar_power" in df.columns:
        df["solar_power"] = pd.to_numeric(df["solar_power"], errors="coerce").fillna(0.0)
        df["solar_kwh_5m"] = np.maximum(df["solar_power"], 0.0) / 12.0 / 1000.0
    else:
        df["solar_kwh_5m"] = 0.0

    if "load_power" in df.columns:
        df["load_power"] = pd.to_numeric(df["load_power"], errors="coerce").fillna(0.0)
        df["load_kwh_5m"] = np.maximum(df["load_power"], 0.0) / 12.0 / 1000.0
    else:
        # Estimate load from grid + solar + battery discharge
        df["load_kwh_5m"] = df["import_kwh_5m"]

    if scenario.ev_exclusion_enabled:
        df["is_car_charging"] = df.apply(
            lambda row: (
                row["grid_power"] >= scenario.ev_min_power_w
                and time_in_window(
                    row["timestamp_local"], scenario.ev_start, scenario.ev_end
                )
            ),
            axis=1,
        )
    else:
        df["is_car_charging"] = False

    df["car_import_kwh_5m"] = np.where(df["is_car_charging"], df["import_kwh_5m"], 0.0)
    df["non_car_import_kwh_5m"] = np.where(~df["is_car_charging"], df["import_kwh_5m"], 0.0)

    hh = (
        df.groupby("slot_start", as_index=False)
        .agg(
            total_import_kwh=("import_kwh_5m", "sum"),
            non_car_import_kwh=("non_car_import_kwh_5m", "sum"),
            car_import_kwh=("car_import_kwh_5m", "sum"),
            export_kwh=("export_kwh_5m", "sum"),
            battery_charge_from_grid_kwh=("battery_charge_from_grid_kwh_5m", "sum"),
            solar_kwh=("solar_kwh_5m", "sum"),
            load_kwh=("load_kwh_5m", "sum"),
        )
        .sort_values("slot_start")
    )
    hh["date"] = hh["slot_start"].dt.date
    return hh


def trim_date_range(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    out = df.copy()
    if start:
        out = out[out["slot_start"] >= pd.Timestamp(start, tz=LOCAL_TZ)]
    if end:
        out = out[out["slot_start"] < pd.Timestamp(end, tz=LOCAL_TZ) + pd.Timedelta(days=1)]
    return out


def build_extra_profile(slots: pd.Series, extra_daily_kwh: float, start_hhmm: str, end_hhmm: str) -> pd.Series:
    profile = pd.Series(0.0, index=slots.index)
    if extra_daily_kwh <= 0:
        return profile
    slot_df = pd.DataFrame({"slot_start": slots})
    slot_df["date"] = slot_df["slot_start"].dt.date
    slot_df["in_window"] = slot_df["slot_start"].apply(lambda ts: time_in_window(ts, start_hhmm, end_hhmm))
    for _, idx in slot_df.groupby("date").groups.items():
        eligible = slot_df.loc[list(idx)]
        use_idx = eligible.index[eligible["in_window"]].tolist()
        if use_idx:
            profile.loc[use_idx] = extra_daily_kwh / len(use_idx)
    return profile


def calculate_intelligent(hh: pd.DataFrame, tariff: IntelligentTariff, scenario: ScenarioConfig):
    df = hh.copy()
    df["date"] = df["slot_start"].dt.date
    df["extra_kwh"] = build_extra_profile(df["slot_start"], scenario.extra_daily_kwh, scenario.extra_start, scenario.extra_end)
    df["is_offpeak"] = df["slot_start"].apply(lambda ts: time_in_window(ts, tariff.offpeak_start, tariff.offpeak_end))

    daily_rows = []
    total_import_cost_p = total_export_revenue_p = total_sc_p = 0.0

    for day, group in df.groupby("date", sort=True):
        group = group.copy().sort_values("slot_start")
        existing_non_car_offpeak = group.loc[group["is_offpeak"], "non_car_import_kwh"].sum()
        existing_non_car_peak = group.loc[~group["is_offpeak"], "non_car_import_kwh"].sum()
        existing_car_import = group["car_import_kwh"].sum()
        export_kwh = group["export_kwh"].sum()
        extra_kwh = group["extra_kwh"].sum()

        battery_cheap_available = scenario.battery_capacity_kwh
        existing_non_car_total = existing_non_car_offpeak + existing_non_car_peak
        existing_non_car_cheap = min(existing_non_car_total, battery_cheap_available)
        remaining_battery = max(battery_cheap_available - existing_non_car_cheap, 0.0)

        extra_cheap_kwh = min(extra_kwh, remaining_battery)
        extra_expensive_kwh = max(extra_kwh - remaining_battery, 0.0)

        car_cost_p = existing_car_import * tariff.offpeak_import_p_per_kwh
        historical_non_car_cost_p = (
            existing_non_car_offpeak * tariff.offpeak_import_p_per_kwh
            + existing_non_car_peak * tariff.peak_import_p_per_kwh
        )
        extra_cost_p = extra_cheap_kwh * tariff.offpeak_import_p_per_kwh + extra_expensive_kwh * tariff.peak_import_p_per_kwh

        import_cost_p = car_cost_p + historical_non_car_cost_p + extra_cost_p
        export_revenue_p = export_kwh * tariff.export_p_per_kwh
        sc_p = tariff.standing_charge_p_per_day

        total_import_cost_p += import_cost_p
        total_export_revenue_p += export_revenue_p
        total_sc_p += sc_p

        daily_rows.append({
            "date": day,
            "car_import_kwh": round(existing_car_import, 4),
            "existing_non_car_import_offpeak_kwh": round(existing_non_car_offpeak, 4),
            "existing_non_car_import_peak_kwh": round(existing_non_car_peak, 4),
            "existing_non_car_total_kwh": round(existing_non_car_total, 4),
            "battery_start_kwh": round(scenario.battery_capacity_kwh, 4),
            "battery_remaining_after_existing_kwh": round(remaining_battery, 4),
            "extra_kwh": round(extra_kwh, 4),
            "extra_cheap_kwh": round(extra_cheap_kwh, 4),
            "extra_expensive_kwh": round(extra_expensive_kwh, 4),
            "battery_ran_out": bool(extra_expensive_kwh > 0 or existing_non_car_total > scenario.battery_capacity_kwh),
            "export_kwh": round(export_kwh, 4),
            "import_cost_p": round(import_cost_p, 4),
            "export_revenue_p": round(export_revenue_p, 4),
            "standing_charge_p": round(sc_p, 4),
            "net_cost_p": round(import_cost_p + sc_p - export_revenue_p, 4),
        })

    summary = {
        "tariff": "intelligent",
        "import_cost_gbp": round(pence_to_pounds(total_import_cost_p), 2),
        "export_revenue_gbp": round(pence_to_pounds(total_export_revenue_p), 2),
        "standing_charge_gbp": round(pence_to_pounds(total_sc_p), 2),
        "net_cost_gbp": round(pence_to_pounds(total_import_cost_p + total_sc_p - total_export_revenue_p), 2),
    }
    return summary, pd.DataFrame(daily_rows)


def _allocate_flexible_energy_to_cheapest_slots(day_df: pd.DataFrame, flexible_kwh: float, max_kw: float) -> pd.Series:
    alloc = pd.Series(0.0, index=day_df.index)
    if flexible_kwh <= 0:
        return alloc
    slot_cap_kwh = max_kw * 0.5
    remaining = flexible_kwh
    for idx in day_df.sort_values("agile_import_p_per_kwh").index.tolist():
        if remaining <= 1e-9:
            break
        add = min(slot_cap_kwh, remaining)
        alloc.loc[idx] += add
        remaining -= add
    return alloc


def calculate_agile(
    hh: pd.DataFrame, agile_import: pd.DataFrame,
    agile_export: pd.DataFrame, config: AgileConfig,
    scenario: ScenarioConfig,
):
    df = hh.copy()
    df = df.merge(
        agile_import[["slot_start", "price_p_per_kwh"]].rename(columns={"price_p_per_kwh": "agile_import_p_per_kwh"}),
        on="slot_start", how="left"
    ).merge(
        agile_export[["slot_start", "price_p_per_kwh"]].rename(columns={"price_p_per_kwh": "agile_export_p_per_kwh"}),
        on="slot_start", how="left"
    )
    if df["agile_import_p_per_kwh"].isna().any() or df["agile_export_p_per_kwh"].isna().any():
        missing = df["agile_import_p_per_kwh"].isna().sum() + df["agile_export_p_per_kwh"].isna().sum()
        print(f"  Warning: dropping {missing} slots with missing Agile prices")
        df = df.dropna(subset=["agile_import_p_per_kwh", "agile_export_p_per_kwh"]).copy()

    df["date"] = df["slot_start"].dt.date
    df["extra_kwh"] = build_extra_profile(df["slot_start"], scenario.extra_daily_kwh, scenario.extra_start, scenario.extra_end)

    daily_rows = []
    total_import_cost_p = total_export_revenue_p = total_sc_p = 0.0
    flexible_daily_cap_kwh = config.flexible_charge_hours_per_day * config.flexible_max_kw

    for day, group in df.groupby("date", sort=True):
        group = group.copy().sort_values("slot_start")
        base_import_kwh = group["total_import_kwh"].sum()
        export_kwh = group["export_kwh"].sum()
        extra_kwh = group["extra_kwh"].sum()

        flexible_kwh = min(extra_kwh, flexible_daily_cap_kwh)
        stranded_kwh = max(extra_kwh - flexible_daily_cap_kwh, 0.0)

        shifted = _allocate_flexible_energy_to_cheapest_slots(group, flexible_kwh, config.flexible_max_kw)
        direct_cost_p = float((group["total_import_kwh"] * group["agile_import_p_per_kwh"]).sum())
        shifted_cost_p = float((shifted * group["agile_import_p_per_kwh"]).sum())

        stranded_cost_p = 0.0
        if stranded_kwh > 1e-9:
            remaining = stranded_kwh
            for idx in group.index[group["extra_kwh"] > 0].tolist():
                if remaining <= 1e-9:
                    break
                use = min(group.loc[idx, "extra_kwh"], remaining)
                stranded_cost_p += use * group.loc[idx, "agile_import_p_per_kwh"]
                remaining -= use

        import_cost_p = direct_cost_p + shifted_cost_p + stranded_cost_p
        export_revenue_p = float((group["export_kwh"] * group["agile_export_p_per_kwh"]).sum())
        sc_p = config.standing_charge_p_per_day

        total_import_cost_p += import_cost_p
        total_export_revenue_p += export_revenue_p
        total_sc_p += sc_p

        daily_rows.append({
            "date": day,
            "base_import_kwh": round(base_import_kwh, 4),
            "export_kwh": round(export_kwh, 4),
            "extra_kwh": round(extra_kwh, 4),
            "flexible_shifted_kwh": round(flexible_kwh, 4),
            "stranded_in_original_window_kwh": round(stranded_kwh, 4),
            "battery_ran_out": bool(stranded_kwh > 0),
            "avg_agile_import_p_per_kwh": round(float(group["agile_import_p_per_kwh"].mean()), 4),
            "avg_agile_export_p_per_kwh": round(float(group["agile_export_p_per_kwh"].mean()), 4),
            "import_cost_p": round(import_cost_p, 4),
            "export_revenue_p": round(export_revenue_p, 4),
            "standing_charge_p": round(sc_p, 4),
            "net_cost_p": round(import_cost_p + sc_p - export_revenue_p, 4),
        })

    summary = {
        "tariff": "agile",
        "import_cost_gbp": round(pence_to_pounds(total_import_cost_p), 2),
        "export_revenue_gbp": round(pence_to_pounds(total_export_revenue_p), 2),
        "standing_charge_gbp": round(pence_to_pounds(total_sc_p), 2),
        "net_cost_gbp": round(pence_to_pounds(total_import_cost_p + total_sc_p - total_export_revenue_p), 2),
    }
    return summary, pd.DataFrame(daily_rows)


def calculate_go(hh: pd.DataFrame, tariff: GoTariff, scenario: ScenarioConfig):
    """Calculate costs on Octopus Go tariff (cheap overnight, flat peak)."""
    df = hh.copy()
    df["date"] = df["slot_start"].dt.date
    df["extra_kwh"] = build_extra_profile(df["slot_start"], scenario.extra_daily_kwh, scenario.extra_start, scenario.extra_end)
    df["is_offpeak"] = df["slot_start"].apply(lambda ts: time_in_window(ts, tariff.offpeak_start, tariff.offpeak_end))

    daily_rows = []
    total_import_cost_p = total_export_revenue_p = total_sc_p = 0.0

    for day, group in df.groupby("date", sort=True):
        group = group.copy().sort_values("slot_start")
        offpeak_import = group.loc[group["is_offpeak"], "total_import_kwh"].sum()
        peak_import = group.loc[~group["is_offpeak"], "total_import_kwh"].sum()
        export_kwh = group["export_kwh"].sum()
        extra_kwh = group["extra_kwh"].sum()

        # Extra consumption: battery covers cheap slots first
        battery_cheap_available = scenario.battery_capacity_kwh
        total_non_car = offpeak_import + peak_import
        cheap_covered = min(total_non_car, battery_cheap_available)
        remaining_battery = max(battery_cheap_available - cheap_covered, 0.0)
        extra_cheap_kwh = min(extra_kwh, remaining_battery)
        extra_expensive_kwh = max(extra_kwh - remaining_battery, 0.0)

        import_cost_p = (
            offpeak_import * tariff.offpeak_import_p_per_kwh
            + peak_import * tariff.peak_import_p_per_kwh
            + extra_cheap_kwh * tariff.offpeak_import_p_per_kwh
            + extra_expensive_kwh * tariff.peak_import_p_per_kwh
        )
        export_revenue_p = export_kwh * tariff.export_p_per_kwh
        sc_p = tariff.standing_charge_p_per_day

        total_import_cost_p += import_cost_p
        total_export_revenue_p += export_revenue_p
        total_sc_p += sc_p

        daily_rows.append({
            "date": day,
            "offpeak_import_kwh": round(offpeak_import, 4),
            "peak_import_kwh": round(peak_import, 4),
            "extra_kwh": round(extra_kwh, 4),
            "extra_cheap_kwh": round(extra_cheap_kwh, 4),
            "extra_expensive_kwh": round(extra_expensive_kwh, 4),
            "export_kwh": round(export_kwh, 4),
            "import_cost_p": round(import_cost_p, 4),
            "export_revenue_p": round(export_revenue_p, 4),
            "standing_charge_p": round(sc_p, 4),
            "net_cost_p": round(import_cost_p + sc_p - export_revenue_p, 4),
        })

    summary = {
        "tariff": "go",
        "import_cost_gbp": round(pence_to_pounds(total_import_cost_p), 2),
        "export_revenue_gbp": round(pence_to_pounds(total_export_revenue_p), 2),
        "standing_charge_gbp": round(pence_to_pounds(total_sc_p), 2),
        "net_cost_gbp": round(pence_to_pounds(total_import_cost_p + total_sc_p - total_export_revenue_p), 2),
    }
    return summary, pd.DataFrame(daily_rows)


def _flux_band(ts: pd.Timestamp, tariff: FluxTariff) -> str:
    """Determine which Flux time band a timestamp falls into."""
    if time_in_window(ts, tariff.offpeak_start, tariff.offpeak_end):
        return "offpeak"
    if time_in_window(ts, tariff.peak_start, tariff.peak_end):
        return "peak"
    return "day"


def calculate_flux(hh: pd.DataFrame, tariff: FluxTariff, scenario: ScenarioConfig):
    """Calculate costs on Octopus Flux tariff (3 bands for import and export)."""
    df = hh.copy()
    df["date"] = df["slot_start"].dt.date
    df["extra_kwh"] = build_extra_profile(df["slot_start"], scenario.extra_daily_kwh, scenario.extra_start, scenario.extra_end)
    df["band"] = df["slot_start"].apply(lambda ts: _flux_band(ts, tariff))

    import_rates = {"offpeak": tariff.offpeak_import_p_per_kwh, "day": tariff.day_import_p_per_kwh, "peak": tariff.peak_import_p_per_kwh}
    export_rates = {"offpeak": tariff.offpeak_export_p_per_kwh, "day": tariff.day_export_p_per_kwh, "peak": tariff.peak_export_p_per_kwh}

    daily_rows = []
    total_import_cost_p = total_export_revenue_p = total_sc_p = 0.0

    for day, group in df.groupby("date", sort=True):
        group = group.copy().sort_values("slot_start")

        import_cost_p = 0.0
        export_revenue_p = 0.0
        offpeak_import = day_import = peak_import = 0.0
        extra_kwh = group["extra_kwh"].sum()

        for band_name in ("offpeak", "day", "peak"):
            band_mask = group["band"] == band_name
            band_import = group.loc[band_mask, "total_import_kwh"].sum()
            band_export = group.loc[band_mask, "export_kwh"].sum()
            import_cost_p += band_import * import_rates[band_name]
            export_revenue_p += band_export * export_rates[band_name]
            if band_name == "offpeak":
                offpeak_import = band_import
            elif band_name == "day":
                day_import = band_import
            else:
                peak_import = band_import

        # Extra consumption charged at cheapest available rate (offpeak if battery covers it)
        battery_cheap_available = scenario.battery_capacity_kwh
        total_existing = offpeak_import + day_import + peak_import
        remaining_battery = max(battery_cheap_available - total_existing, 0.0)
        extra_cheap_kwh = min(extra_kwh, remaining_battery)
        extra_expensive_kwh = max(extra_kwh - remaining_battery, 0.0)
        import_cost_p += extra_cheap_kwh * tariff.offpeak_import_p_per_kwh + extra_expensive_kwh * tariff.day_import_p_per_kwh

        sc_p = tariff.standing_charge_p_per_day
        total_import_cost_p += import_cost_p
        total_export_revenue_p += export_revenue_p
        total_sc_p += sc_p

        daily_rows.append({
            "date": day,
            "offpeak_import_kwh": round(offpeak_import, 4),
            "day_import_kwh": round(day_import, 4),
            "peak_import_kwh": round(peak_import, 4),
            "extra_kwh": round(extra_kwh, 4),
            "export_kwh": round(group["export_kwh"].sum(), 4),
            "import_cost_p": round(import_cost_p, 4),
            "export_revenue_p": round(export_revenue_p, 4),
            "standing_charge_p": round(sc_p, 4),
            "net_cost_p": round(import_cost_p + sc_p - export_revenue_p, 4),
        })

    summary = {
        "tariff": "flux",
        "import_cost_gbp": round(pence_to_pounds(total_import_cost_p), 2),
        "export_revenue_gbp": round(pence_to_pounds(total_export_revenue_p), 2),
        "standing_charge_gbp": round(pence_to_pounds(total_sc_p), 2),
        "net_cost_gbp": round(pence_to_pounds(total_import_cost_p + total_sc_p - total_export_revenue_p), 2),
    }
    return summary, pd.DataFrame(daily_rows)


def _cosy_band(ts: pd.Timestamp, tariff: CosyTariff) -> str:
    """Determine which Cosy time band a timestamp falls into."""
    if time_in_window(ts, tariff.peak_start, tariff.peak_end):
        return "peak"
    if (time_in_window(ts, tariff.cosy_window_1_start, tariff.cosy_window_1_end)
            or time_in_window(ts, tariff.cosy_window_2_start, tariff.cosy_window_2_end)
            or time_in_window(ts, tariff.cosy_window_3_start, tariff.cosy_window_3_end)):
        return "cosy"
    return "day"


def calculate_cosy(hh: pd.DataFrame, tariff: CosyTariff, scenario: ScenarioConfig):
    """Calculate costs on Octopus Cosy tariff (3 cheap windows, peak, standard)."""
    df = hh.copy()
    df["date"] = df["slot_start"].dt.date
    df["extra_kwh"] = build_extra_profile(df["slot_start"], scenario.extra_daily_kwh, scenario.extra_start, scenario.extra_end)
    df["band"] = df["slot_start"].apply(lambda ts: _cosy_band(ts, tariff))

    import_rates = {"cosy": tariff.cosy_import_p_per_kwh, "day": tariff.day_import_p_per_kwh, "peak": tariff.peak_import_p_per_kwh}

    daily_rows = []
    total_import_cost_p = total_export_revenue_p = total_sc_p = 0.0

    for day, group in df.groupby("date", sort=True):
        group = group.copy().sort_values("slot_start")

        import_cost_p = 0.0
        cosy_import = day_import = peak_import = 0.0

        for band_name in ("cosy", "day", "peak"):
            band_mask = group["band"] == band_name
            band_import = group.loc[band_mask, "total_import_kwh"].sum()
            import_cost_p += band_import * import_rates[band_name]
            if band_name == "cosy":
                cosy_import = band_import
            elif band_name == "day":
                day_import = band_import
            else:
                peak_import = band_import

        export_kwh = group["export_kwh"].sum()
        extra_kwh = group["extra_kwh"].sum()

        # Extra consumption: battery covers cheap slots first
        battery_cheap_available = scenario.battery_capacity_kwh
        total_existing = cosy_import + day_import + peak_import
        remaining_battery = max(battery_cheap_available - total_existing, 0.0)
        extra_cheap_kwh = min(extra_kwh, remaining_battery)
        extra_expensive_kwh = max(extra_kwh - remaining_battery, 0.0)
        import_cost_p += extra_cheap_kwh * tariff.cosy_import_p_per_kwh + extra_expensive_kwh * tariff.day_import_p_per_kwh

        export_revenue_p = export_kwh * tariff.export_p_per_kwh
        sc_p = tariff.standing_charge_p_per_day

        total_import_cost_p += import_cost_p
        total_export_revenue_p += export_revenue_p
        total_sc_p += sc_p

        daily_rows.append({
            "date": day,
            "cosy_import_kwh": round(cosy_import, 4),
            "day_import_kwh": round(day_import, 4),
            "peak_import_kwh": round(peak_import, 4),
            "extra_kwh": round(extra_kwh, 4),
            "export_kwh": round(export_kwh, 4),
            "import_cost_p": round(import_cost_p, 4),
            "export_revenue_p": round(export_revenue_p, 4),
            "standing_charge_p": round(sc_p, 4),
            "net_cost_p": round(import_cost_p + sc_p - export_revenue_p, 4),
        })

    summary = {
        "tariff": "cosy",
        "import_cost_gbp": round(pence_to_pounds(total_import_cost_p), 2),
        "export_revenue_gbp": round(pence_to_pounds(total_export_revenue_p), 2),
        "standing_charge_gbp": round(pence_to_pounds(total_sc_p), 2),
        "net_cost_gbp": round(pence_to_pounds(total_import_cost_p + total_sc_p - total_export_revenue_p), 2),
    }
    return summary, pd.DataFrame(daily_rows)


def calculate_flexible(hh: pd.DataFrame, tariff: FlexibleTariff, scenario: ScenarioConfig):
    """Calculate costs on Octopus Flexible (SVR) — flat rate tariff."""
    df = hh.copy()
    df["date"] = df["slot_start"].dt.date
    df["extra_kwh"] = build_extra_profile(df["slot_start"], scenario.extra_daily_kwh, scenario.extra_start, scenario.extra_end)

    daily_rows = []
    total_import_cost_p = total_export_revenue_p = total_sc_p = 0.0

    for day, group in df.groupby("date", sort=True):
        total_import = group["total_import_kwh"].sum()
        export_kwh = group["export_kwh"].sum()
        extra_kwh = group["extra_kwh"].sum()

        import_cost_p = (total_import + extra_kwh) * tariff.import_p_per_kwh
        export_revenue_p = export_kwh * tariff.export_p_per_kwh
        sc_p = tariff.standing_charge_p_per_day

        total_import_cost_p += import_cost_p
        total_export_revenue_p += export_revenue_p
        total_sc_p += sc_p

        daily_rows.append({
            "date": day,
            "total_import_kwh": round(total_import, 4),
            "extra_kwh": round(extra_kwh, 4),
            "export_kwh": round(export_kwh, 4),
            "import_cost_p": round(import_cost_p, 4),
            "export_revenue_p": round(export_revenue_p, 4),
            "standing_charge_p": round(sc_p, 4),
            "net_cost_p": round(import_cost_p + sc_p - export_revenue_p, 4),
        })

    summary = {
        "tariff": "flexible",
        "import_cost_gbp": round(pence_to_pounds(total_import_cost_p), 2),
        "export_revenue_gbp": round(pence_to_pounds(total_export_revenue_p), 2),
        "standing_charge_gbp": round(pence_to_pounds(total_sc_p), 2),
        "net_cost_gbp": round(pence_to_pounds(total_import_cost_p + total_sc_p - total_export_revenue_p), 2),
    }
    return summary, pd.DataFrame(daily_rows)


def download_tracker_rates(region_code: str, cache_dir: Path, force: bool = False) -> Optional[pd.DataFrame]:
    """Download Octopus Tracker daily rates from the Octopus API.

    Returns a DataFrame with columns: date, price_p_per_kwh
    Returns None if rates cannot be fetched.
    """
    region_code = region_code.upper()
    product_code = "SILVER-24-04-03"  # Current Tracker product code
    tariff_code = f"E-1R-{product_code}-{region_code}"
    cache_path = cache_dir / f"tracker_{region_code}.csv"

    if not force and cache_path.exists():
        df = pd.read_csv(cache_path)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    url = f"https://api.octopus.energy/v1/products/{product_code}/electricity-tariffs/{tariff_code}/standard-unit-rates/"
    all_results = []
    page_url = url

    try:
        while page_url:
            resp = requests.get(page_url, timeout=30, params={"page_size": 1500})
            resp.raise_for_status()
            data = resp.json()
            all_results.extend(data.get("results", []))
            page_url = data.get("next")
    except requests.RequestException as e:
        print(f"  Warning: could not fetch Tracker rates: {e}")
        # Try fallback product code
        product_code = "SILVER-FLEX-22-11-25"
        tariff_code = f"E-1R-{product_code}-{region_code}"
        url = f"https://api.octopus.energy/v1/products/{product_code}/electricity-tariffs/{tariff_code}/standard-unit-rates/"
        page_url = url
        all_results = []
        try:
            while page_url:
                resp = requests.get(page_url, timeout=30, params={"page_size": 1500})
                resp.raise_for_status()
                data = resp.json()
                all_results.extend(data.get("results", []))
                page_url = data.get("next")
        except requests.RequestException as e2:
            print(f"  Warning: fallback Tracker fetch also failed: {e2}")
            return None

    if not all_results:
        print("  Warning: no Tracker rate data returned from API")
        return None

    df = pd.DataFrame(all_results)
    df["valid_from"] = pd.to_datetime(df["valid_from"], utc=True)
    df["date"] = df["valid_from"].dt.tz_convert(LOCAL_TZ).dt.date
    # Tracker has one rate per day — take the last one per date if duplicates
    df = df.sort_values("valid_from").drop_duplicates(subset=["date"], keep="last")
    df = df.rename(columns={"value_inc_vat": "price_p_per_kwh"})
    df = df[["date", "price_p_per_kwh"]].sort_values("date")

    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


def calculate_tracker(hh: pd.DataFrame, tracker_rates: pd.DataFrame, tariff: TrackerTariff, scenario: ScenarioConfig):
    """Calculate costs on Octopus Tracker tariff (daily variable rate)."""
    df = hh.copy()
    df["date"] = df["slot_start"].dt.date
    df["extra_kwh"] = build_extra_profile(df["slot_start"], scenario.extra_daily_kwh, scenario.extra_start, scenario.extra_end)

    # Merge tracker daily rates
    tracker_rates = tracker_rates.copy()
    tracker_rates["date"] = pd.to_datetime(tracker_rates["date"]).dt.date
    df = df.merge(tracker_rates[["date", "price_p_per_kwh"]], on="date", how="left")

    if df["price_p_per_kwh"].isna().any():
        missing_days = df[df["price_p_per_kwh"].isna()]["date"].nunique()
        print(f"  Warning: dropping {missing_days} days with missing Tracker prices")
        df = df.dropna(subset=["price_p_per_kwh"]).copy()

    daily_rows = []
    total_import_cost_p = total_export_revenue_p = total_sc_p = 0.0

    for day, group in df.groupby("date", sort=True):
        total_import = group["total_import_kwh"].sum()
        export_kwh = group["export_kwh"].sum()
        extra_kwh = group["extra_kwh"].sum()
        day_rate = group["price_p_per_kwh"].iloc[0]

        import_cost_p = (total_import + extra_kwh) * day_rate
        export_revenue_p = export_kwh * tariff.export_p_per_kwh
        sc_p = tariff.standing_charge_p_per_day

        total_import_cost_p += import_cost_p
        total_export_revenue_p += export_revenue_p
        total_sc_p += sc_p

        daily_rows.append({
            "date": day,
            "total_import_kwh": round(total_import, 4),
            "extra_kwh": round(extra_kwh, 4),
            "export_kwh": round(export_kwh, 4),
            "tracker_rate_p_per_kwh": round(day_rate, 4),
            "import_cost_p": round(import_cost_p, 4),
            "export_revenue_p": round(export_revenue_p, 4),
            "standing_charge_p": round(sc_p, 4),
            "net_cost_p": round(import_cost_p + sc_p - export_revenue_p, 4),
        })

    summary = {
        "tariff": "tracker",
        "import_cost_gbp": round(pence_to_pounds(total_import_cost_p), 2),
        "export_revenue_gbp": round(pence_to_pounds(total_export_revenue_p), 2),
        "standing_charge_gbp": round(pence_to_pounds(total_sc_p), 2),
        "net_cost_gbp": round(pence_to_pounds(total_import_cost_p + total_sc_p - total_export_revenue_p), 2),
    }
    return summary, pd.DataFrame(daily_rows)


# ── Optimised Battery Charging ────────────────────────────────────────────────


def _get_slot_rate_for_tariff(slot_start: pd.Timestamp, tariff_name: str, tariff_config, agile_rates=None, tracker_rates=None):
    """Get the import rate for a given slot based on tariff type."""
    if tariff_name == "intelligent":
        if time_in_window(slot_start, tariff_config.offpeak_start, tariff_config.offpeak_end):
            return tariff_config.offpeak_import_p_per_kwh
        return tariff_config.peak_import_p_per_kwh
    if tariff_name == "go":
        if time_in_window(slot_start, tariff_config.offpeak_start, tariff_config.offpeak_end):
            return tariff_config.offpeak_import_p_per_kwh
        return tariff_config.peak_import_p_per_kwh
    if tariff_name == "flux":
        band = _flux_band(slot_start, tariff_config)
        rates = {
            "offpeak": tariff_config.offpeak_import_p_per_kwh,
            "day": tariff_config.day_import_p_per_kwh,
            "peak": tariff_config.peak_import_p_per_kwh,
        }
        return rates[band]
    if tariff_name == "cosy":
        band = _cosy_band(slot_start, tariff_config)
        rates = {
            "cosy": tariff_config.cosy_import_p_per_kwh,
            "day": tariff_config.day_import_p_per_kwh,
            "peak": tariff_config.peak_import_p_per_kwh,
        }
        return rates[band]
    if tariff_name == "flexible":
        return tariff_config.import_p_per_kwh
    if tariff_name == "agile" and agile_rates is not None:
        match = agile_rates[agile_rates["slot_start"] == slot_start]
        if not match.empty:
            return float(match.iloc[0]["price_p_per_kwh"])
        return None
    if tariff_name == "tracker" and tracker_rates is not None:
        day = slot_start.date()
        match = tracker_rates[tracker_rates["date"] == day]
        if not match.empty:
            return float(match.iloc[0]["price_p_per_kwh"])
        return None
    return None


def calculate_optimised_charging(
    hh: pd.DataFrame,
    tariff_name: str,
    tariff_config,
    scenario: ScenarioConfig,
    agile_rates=None,
    tracker_rates=None,
    battery_max_charge_kw: float = 5.0,
):
    """Calculate savings from moving battery charging to cheapest slots.

    For each day:
    1. Identify how much the battery charged from the grid (and at what cost)
    2. Remove that charging from its original slots
    3. Re-allocate it to the cheapest available slots (respecting max charge rate)
    4. Report actual cost vs optimised cost and the saving

    Args:
        hh: Half-hourly power data with battery_charge_from_grid_kwh column
        tariff_name: Name of the tariff being analysed
        tariff_config: Tariff configuration dataclass
        scenario: Scenario configuration
        agile_rates: Agile import rates DataFrame (for agile tariff)
        tracker_rates: Tracker daily rates DataFrame (for tracker tariff)
        battery_max_charge_kw: Maximum battery charge rate in kW (default 5.0 for Powerwall 2)
    """
    df = hh.copy()
    df["date"] = df["slot_start"].dt.date

    # Get the import rate for each slot
    df["import_rate_p"] = df["slot_start"].apply(
        lambda ts: _get_slot_rate_for_tariff(ts, tariff_name, tariff_config, agile_rates, tracker_rates)
    )

    # Drop slots with no rate data (e.g. missing agile/tracker prices)
    df = df.dropna(subset=["import_rate_p"]).copy()

    if df.empty:
        return None, None

    slot_max_kwh = battery_max_charge_kw * 0.5  # max kWh per 30-min slot

    daily_rows = []
    total_actual_cost_p = 0.0
    total_optimised_cost_p = 0.0
    total_battery_kwh = 0.0

    for day, group in df.groupby("date", sort=True):
        group = group.copy().sort_values("slot_start")
        daily_battery_kwh = group["battery_charge_from_grid_kwh"].sum()

        if daily_battery_kwh < 0.01:
            # No meaningful battery charging this day
            daily_rows.append({
                "date": day,
                "battery_charge_kwh": 0.0,
                "actual_cost_p": 0.0,
                "optimised_cost_p": 0.0,
                "saving_p": 0.0,
                "cheapest_window": "",
            })
            continue

        # Actual cost: what was paid for battery charging in original slots
        actual_cost_p = float((group["battery_charge_from_grid_kwh"] * group["import_rate_p"]).sum())

        # Optimised: allocate the same kWh to cheapest slots
        sorted_by_price = group.sort_values("import_rate_p")
        remaining_kwh = daily_battery_kwh
        optimised_cost_p = 0.0
        cheapest_slots = []

        for _, row in sorted_by_price.iterrows():
            if remaining_kwh <= 1e-9:
                break
            allocate = min(slot_max_kwh, remaining_kwh)
            optimised_cost_p += allocate * row["import_rate_p"]
            remaining_kwh -= allocate
            cheapest_slots.append(row["slot_start"].strftime("%H:%M"))

        saving_p = actual_cost_p - optimised_cost_p

        total_actual_cost_p += actual_cost_p
        total_optimised_cost_p += optimised_cost_p
        total_battery_kwh += daily_battery_kwh

        # Show the time window used for optimised charging
        if cheapest_slots:
            window_start = cheapest_slots[0]
            window_end = cheapest_slots[-1]
            cheapest_window = f"{window_start}-{window_end}" if window_start != window_end else window_start
        else:
            cheapest_window = ""

        daily_rows.append({
            "date": day,
            "battery_charge_kwh": round(daily_battery_kwh, 4),
            "actual_cost_p": round(actual_cost_p, 4),
            "optimised_cost_p": round(optimised_cost_p, 4),
            "saving_p": round(saving_p, 4),
            "cheapest_window": cheapest_window,
        })

    total_saving_p = total_actual_cost_p - total_optimised_cost_p
    summary = {
        "tariff": tariff_name,
        "total_battery_charge_kwh": round(total_battery_kwh, 2),
        "actual_charging_cost_gbp": round(pence_to_pounds(total_actual_cost_p), 2),
        "optimised_charging_cost_gbp": round(pence_to_pounds(total_optimised_cost_p), 2),
        "saving_gbp": round(pence_to_pounds(total_saving_p), 2),
        "saving_pct": round(total_saving_p / total_actual_cost_p * 100, 1) if total_actual_cost_p > 0 else 0.0,
    }
    return summary, pd.DataFrame(daily_rows)


# ── Best Tariff Full Simulation ───────────────────────────────────────────────


def _get_export_rate_for_tariff(slot_start: pd.Timestamp, tariff_name: str, tariff_config, agile_export_rates=None):
    """Get the export rate for a given slot based on tariff type."""
    if tariff_name == "flux":
        band = _flux_band(slot_start, tariff_config)
        rates = {
            "offpeak": tariff_config.offpeak_export_p_per_kwh,
            "day": tariff_config.day_export_p_per_kwh,
            "peak": tariff_config.peak_export_p_per_kwh,
        }
        return rates[band]
    if tariff_name == "agile" and agile_export_rates is not None:
        match = agile_export_rates[agile_export_rates["slot_start"] == slot_start]
        if not match.empty:
            return float(match.iloc[0]["price_p_per_kwh"])
        return None
    # All other tariffs have a flat export rate
    if hasattr(tariff_config, "export_p_per_kwh"):
        return tariff_config.export_p_per_kwh
    return 12.0  # fallback


def simulate_optimal_battery(
    hh: pd.DataFrame,
    tariff_name: str,
    tariff_config,
    scenario: ScenarioConfig,
    agile_import_rates=None,
    agile_export_rates=None,
    tracker_rates=None,
    battery_max_charge_kw: float = 5.0,
    battery_max_discharge_kw: float = 5.0,
    battery_efficiency: float = 0.90,
):
    """Simulate optimal battery usage for a tariff using full energy flow.

    For each day, runs a two-pass optimisation:
    Pass 1: Determine import/export rates for all slots
    Pass 2: Simulate battery operation slot-by-slot with optimal strategy:
      - Solar first serves load, surplus charges battery or exports
      - Battery charges from grid during cheapest import slots
      - Battery discharges to avoid expensive imports or to export at peak prices
      - Respects battery capacity, charge/discharge rate limits, and round-trip efficiency

    Returns both the "actual" cost (what really happened) and the "optimised" cost
    (what would happen with perfect battery scheduling).
    """
    df = hh.copy()
    df["date"] = df["slot_start"].dt.date

    # Get import and export rates for each slot
    df["import_rate_p"] = df["slot_start"].apply(
        lambda ts: _get_slot_rate_for_tariff(ts, tariff_name, tariff_config, agile_import_rates, tracker_rates)
    )
    df["export_rate_p"] = df["slot_start"].apply(
        lambda ts: _get_export_rate_for_tariff(ts, tariff_name, tariff_config, agile_export_rates)
    )

    # Drop slots with missing rate data
    df = df.dropna(subset=["import_rate_p", "export_rate_p"]).copy()
    if df.empty:
        return None, None

    slot_max_charge_kwh = battery_max_charge_kw * 0.5
    slot_max_discharge_kwh = battery_max_discharge_kw * 0.5
    battery_cap = scenario.battery_capacity_kwh

    daily_rows = []
    total_actual_import_cost_p = 0.0
    total_actual_export_revenue_p = 0.0
    total_opt_import_cost_p = 0.0
    total_opt_export_revenue_p = 0.0
    total_days = 0

    for day, group in df.groupby("date", sort=True):
        group = group.copy().sort_values("slot_start").reset_index(drop=True)
        n_slots = len(group)
        if n_slots == 0:
            continue
        total_days += 1

        # ── Actual costs (what really happened) ──
        actual_import_cost_p = float((group["total_import_kwh"] * group["import_rate_p"]).sum())
        actual_export_revenue_p = float((group["export_kwh"] * group["export_rate_p"]).sum())

        # ── Optimised simulation ──
        # Strategy: for each slot, decide battery action based on rates
        # We use a greedy approach:
        # 1. Solar serves load first (free)
        # 2. Surplus solar charges battery (free), then exports
        # 3. When load > solar: discharge battery if import rate is high,
        #    or import from grid if rate is cheap
        # 4. Charge battery from grid during cheapest slots if there's
        #    capacity and it's worth it (cheap import can offset future expensive import/enable peak export)

        # First, determine the "value" of stored energy for each slot:
        # It's worth charging if we can later discharge at a higher rate
        # Simple heuristic: charge when import rate is below daily median,
        # discharge when import rate is above median or export rate is attractive

        solar = group["solar_kwh"].values
        load = group["load_kwh"].values
        import_rates = group["import_rate_p"].values
        export_rates = group["export_rate_p"].values

        # Sort slots by import rate to identify cheap/expensive periods
        sorted_rates = np.sort(import_rates)

        # Determine threshold: charge from grid when rate is below this
        # Discharge/avoid import when rate is above this
        # For time-of-use tariffs, this naturally separates off-peak from peak
        # For variable tariffs, it finds the daily sweet spot
        sorted_rates = np.sort(import_rates)
        # Use the rate at the point where we'd fill the battery as threshold
        slots_to_fill = int(np.ceil(battery_cap / slot_max_charge_kwh))
        if slots_to_fill < n_slots:
            charge_threshold = sorted_rates[min(slots_to_fill, n_slots - 1)]
        else:
            charge_threshold = sorted_rates[-1]

        # Also consider export opportunity: discharge if export rate > import rate
        # (arbitrage opportunity on Flux/Agile)

        # Simulate slot by slot
        soc = battery_cap * 0.5  # Assume battery starts at 50% (reasonable daily average)
        opt_import_cost_p = 0.0
        opt_export_revenue_p = 0.0
        opt_grid_import_kwh = 0.0
        opt_grid_export_kwh = 0.0
        opt_battery_charged_kwh = 0.0
        opt_battery_discharged_kwh = 0.0

        for i in range(n_slots):
            slot_solar = solar[i]
            slot_load = load[i]
            slot_import_rate = import_rates[i]
            slot_export_rate = export_rates[i]

            # Step 1: Solar serves load
            solar_to_load = min(slot_solar, slot_load)
            remaining_solar = slot_solar - solar_to_load
            remaining_load = slot_load - solar_to_load

            # Step 2: Surplus solar → battery or export
            solar_to_battery = 0.0
            solar_to_export = 0.0
            if remaining_solar > 0:
                can_charge = min(remaining_solar, slot_max_charge_kwh, battery_cap - soc)
                solar_to_battery = can_charge
                soc += solar_to_battery * battery_efficiency
                solar_to_export = remaining_solar - solar_to_battery

            # Step 3: Remaining load — battery discharge or grid import?
            battery_to_load = 0.0
            grid_to_load = 0.0
            if remaining_load > 0:
                # Discharge battery if import rate is expensive (above threshold)
                # or if battery is full and we'd waste solar tomorrow
                if slot_import_rate >= charge_threshold and soc > 0:
                    can_discharge = min(remaining_load, slot_max_discharge_kwh, soc)
                    battery_to_load = can_discharge
                    soc -= battery_to_load
                    remaining_load -= battery_to_load

                grid_to_load = remaining_load

            # Step 4: Should we charge from grid? (cheap slot, battery not full)
            grid_to_battery = 0.0
            if slot_import_rate < charge_threshold and soc < battery_cap:
                available_charge = min(
                    slot_max_charge_kwh - solar_to_battery,  # remaining charge capacity this slot
                    (battery_cap - soc) / battery_efficiency,  # space in battery (accounting for losses)
                )
                if available_charge > 0.01:
                    grid_to_battery = available_charge
                    soc += grid_to_battery * battery_efficiency

            # Step 5: Should we discharge to export? (high export rate, worth it)
            battery_to_export = 0.0
            # Only export from battery if export rate exceeds what we'd pay to refill
            if slot_export_rate > charge_threshold and soc > battery_cap * 0.2:
                # Don't drain below 20% — keep reserve for evening load
                available_discharge = min(
                    slot_max_discharge_kwh - battery_to_load,
                    soc - battery_cap * 0.2,
                )
                if available_discharge > 0.01:
                    battery_to_export = available_discharge
                    soc -= battery_to_export

            # Clamp SoC
            soc = max(0.0, min(soc, battery_cap))

            # Calculate costs for this slot
            total_grid_import = grid_to_load + grid_to_battery
            total_grid_export = solar_to_export + battery_to_export

            opt_import_cost_p += total_grid_import * slot_import_rate
            opt_export_revenue_p += total_grid_export * slot_export_rate
            opt_grid_import_kwh += total_grid_import
            opt_grid_export_kwh += total_grid_export
            opt_battery_charged_kwh += solar_to_battery + grid_to_battery
            opt_battery_discharged_kwh += battery_to_load + battery_to_export

        total_actual_import_cost_p += actual_import_cost_p
        total_actual_export_revenue_p += actual_export_revenue_p
        total_opt_import_cost_p += opt_import_cost_p
        total_opt_export_revenue_p += opt_export_revenue_p

        actual_net_p = actual_import_cost_p - actual_export_revenue_p
        opt_net_p = opt_import_cost_p - opt_export_revenue_p

        daily_rows.append({
            "date": day,
            "actual_import_cost_p": round(actual_import_cost_p, 2),
            "actual_export_revenue_p": round(actual_export_revenue_p, 2),
            "actual_net_cost_p": round(actual_net_p, 2),
            "optimised_import_cost_p": round(opt_import_cost_p, 2),
            "optimised_export_revenue_p": round(opt_export_revenue_p, 2),
            "optimised_net_cost_p": round(opt_net_p, 2),
            "saving_p": round(actual_net_p - opt_net_p, 2),
            "opt_grid_import_kwh": round(opt_grid_import_kwh, 2),
            "opt_grid_export_kwh": round(opt_grid_export_kwh, 2),
            "opt_battery_cycles_kwh": round(opt_battery_charged_kwh, 2),
        })

    if total_days == 0:
        return None, None

    # Get standing charge
    sc_p_per_day = 53.35  # default
    if hasattr(tariff_config, "standing_charge_p_per_day"):
        sc_p_per_day = tariff_config.standing_charge_p_per_day
    total_sc_p = sc_p_per_day * total_days

    actual_net_total_p = total_actual_import_cost_p - total_actual_export_revenue_p + total_sc_p
    opt_net_total_p = total_opt_import_cost_p - total_opt_export_revenue_p + total_sc_p
    saving_p = actual_net_total_p - opt_net_total_p

    summary = {
        "tariff": tariff_name,
        "days": total_days,
        "actual_import_gbp": round(pence_to_pounds(total_actual_import_cost_p), 2),
        "actual_export_gbp": round(pence_to_pounds(total_actual_export_revenue_p), 2),
        "actual_standing_gbp": round(pence_to_pounds(total_sc_p), 2),
        "actual_net_gbp": round(pence_to_pounds(actual_net_total_p), 2),
        "optimised_import_gbp": round(pence_to_pounds(total_opt_import_cost_p), 2),
        "optimised_export_gbp": round(pence_to_pounds(total_opt_export_revenue_p), 2),
        "optimised_net_gbp": round(pence_to_pounds(opt_net_total_p), 2),
        "saving_gbp": round(pence_to_pounds(saving_p), 2),
        "saving_pct": round(saving_p / actual_net_total_p * 100, 1) if actual_net_total_p > 0 else 0.0,
    }
    return summary, pd.DataFrame(daily_rows)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def print_summary_table(rows):
    headers = ["tariff", "import_cost_gbp", "export_revenue_gbp", "standing_charge_gbp", "net_cost_gbp"]
    widths = {h: max(len(h), max(len(str(r.get(h, ""))) for r in rows)) for h in headers}
    print("  ".join(h.ljust(widths[h]) for h in headers))
    print("  ".join("-" * widths[h] for h in headers))
    for row in rows:
        print("  ".join(str(row.get(h, "")).ljust(widths[h]) for h in headers))


def run_compare(args) -> int:
    # Determine which tariffs to run
    tariffs_to_run = getattr(args, "tariffs", None)
    all_tariffs = {"intelligent", "agile", "go", "flux", "cosy", "flexible", "tracker"}
    if tariffs_to_run:
        selected = {t.strip().lower() for t in tariffs_to_run.split(",")}
        invalid = selected - all_tariffs
        if invalid:
            print(f"Error: unknown tariff(s): {', '.join(sorted(invalid))}")
            print(f"Available: {', '.join(sorted(all_tariffs))}")
            return 1
    else:
        selected = all_tariffs

    scenario = ScenarioConfig(
        battery_capacity_kwh=args.battery_capacity_kwh,
        extra_daily_kwh=args.extra_daily_kwh,
        extra_start=args.extra_start,
        extra_end=args.extra_end,
        ev_exclusion_enabled=not args.no_ev_exclusion,
    )

    hh = load_power_csv(Path(args.power_csv), scenario)
    hh = trim_date_range(hh, args.start_date, args.end_date)
    if hh.empty:
        raise RuntimeError("No power data left after applying date filter")

    summaries = []
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Intelligent
    if "intelligent" in selected:
        intelligent = IntelligentTariff(
            offpeak_import_p_per_kwh=args.intelligent_offpeak_import,
            peak_import_p_per_kwh=args.intelligent_peak_import,
            export_p_per_kwh=args.intelligent_export,
            standing_charge_p_per_day=args.intelligent_standing_charge,
            offpeak_start=args.intelligent_offpeak_start,
            offpeak_end=args.intelligent_offpeak_end,
        )
        s, daily = calculate_intelligent(hh, intelligent, scenario)
        summaries.append(s)
        write_csv(daily, out_dir / "daily_breakdown_intelligent.csv")

    # Agile
    if "agile" in selected:
        agile_cfg = AgileConfig(
            standing_charge_p_per_day=args.agile_standing_charge,
            flexible_charge_hours_per_day=args.agile_flex_hours,
            flexible_max_kw=args.agile_flex_max_kw,
        )
        in_path, out_path = download_region_tariffs(args.region_code, Path(args.cache_dir), force=args.refresh_tariffs)
        agile_in = read_agile_csv(in_path)
        agile_out = read_agile_csv(out_path)
        s, daily = calculate_agile(hh, agile_in, agile_out, agile_cfg, scenario)
        summaries.append(s)
        write_csv(daily, out_dir / "daily_breakdown_agile.csv")

    # Go
    if "go" in selected:
        go = GoTariff(
            offpeak_import_p_per_kwh=getattr(args, "go_offpeak_import", 8.0),
            peak_import_p_per_kwh=getattr(args, "go_peak_import", 24.5),
            export_p_per_kwh=getattr(args, "go_export", 12.0),
            standing_charge_p_per_day=getattr(args, "go_standing_charge", 53.35),
            offpeak_start=getattr(args, "go_offpeak_start", "23:30"),
            offpeak_end=getattr(args, "go_offpeak_end", "05:30"),
        )
        s, daily = calculate_go(hh, go, scenario)
        summaries.append(s)
        write_csv(daily, out_dir / "daily_breakdown_go.csv")

    # Flux
    if "flux" in selected:
        flux = FluxTariff(
            offpeak_import_p_per_kwh=getattr(args, "flux_offpeak_import", 9.80),
            day_import_p_per_kwh=getattr(args, "flux_day_import", 22.36),
            peak_import_p_per_kwh=getattr(args, "flux_peak_import", 33.54),
            offpeak_export_p_per_kwh=getattr(args, "flux_offpeak_export", 4.05),
            day_export_p_per_kwh=getattr(args, "flux_day_export", 14.40),
            peak_export_p_per_kwh=getattr(args, "flux_peak_export", 28.60),
            standing_charge_p_per_day=getattr(args, "flux_standing_charge", 48.93),
            offpeak_start=getattr(args, "flux_offpeak_start", "02:00"),
            offpeak_end=getattr(args, "flux_offpeak_end", "05:00"),
            peak_start=getattr(args, "flux_peak_start", "16:00"),
            peak_end=getattr(args, "flux_peak_end", "19:00"),
        )
        s, daily = calculate_flux(hh, flux, scenario)
        summaries.append(s)
        write_csv(daily, out_dir / "daily_breakdown_flux.csv")

    # Cosy
    if "cosy" in selected:
        cosy = CosyTariff(
            cosy_import_p_per_kwh=getattr(args, "cosy_import", 12.0),
            day_import_p_per_kwh=getattr(args, "cosy_day_import", 24.50),
            peak_import_p_per_kwh=getattr(args, "cosy_peak_import", 36.75),
            export_p_per_kwh=getattr(args, "cosy_export", 12.0),
            standing_charge_p_per_day=getattr(args, "cosy_standing_charge", 53.35),
        )
        s, daily = calculate_cosy(hh, cosy, scenario)
        summaries.append(s)
        write_csv(daily, out_dir / "daily_breakdown_cosy.csv")

    # Flexible (SVR)
    if "flexible" in selected:
        flexible = FlexibleTariff(
            import_p_per_kwh=getattr(args, "flexible_import", 24.50),
            export_p_per_kwh=getattr(args, "flexible_export", 12.0),
            standing_charge_p_per_day=getattr(args, "flexible_standing_charge", 53.35),
        )
        s, daily = calculate_flexible(hh, flexible, scenario)
        summaries.append(s)
        write_csv(daily, out_dir / "daily_breakdown_flexible.csv")

    # Tracker
    if "tracker" in selected:
        tracker_tariff = TrackerTariff(
            standing_charge_p_per_day=getattr(args, "tracker_standing_charge", 53.35),
            export_p_per_kwh=getattr(args, "tracker_export", 12.0),
        )
        tracker_rates = download_tracker_rates(args.region_code, Path(args.cache_dir), force=args.refresh_tariffs)
        if tracker_rates is not None and not tracker_rates.empty:
            s, daily = calculate_tracker(hh, tracker_rates, tracker_tariff, scenario)
            summaries.append(s)
            write_csv(daily, out_dir / "daily_breakdown_tracker.csv")
        else:
            print("  Skipping Tracker: no rate data available")

    if not summaries:
        print("No tariffs calculated.")
        return 1

    summary_df = pd.DataFrame(summaries).sort_values("net_cost_gbp")
    write_csv(summary_df, out_dir / "summary.csv")

    rows = summary_df.to_dict(orient="records")
    print_summary_table(rows)
    print(f"\nBest tariff by model: {rows[0]['tariff']}")
    print(f"Outputs written to: {out_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    saved = load_defaults()

    def d(key, fallback):
        """Return saved default if available, otherwise the hardcoded fallback."""
        return saved.get(key, fallback)

    parser = argparse.ArgumentParser(
        description="Compare Tesla Powerwall usage against Octopus Intelligent and Agile tariffs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  # Save your settings once
  %(prog)s set-defaults --region-code M --battery-capacity-kwh 13 --email you@example.com

  # Download a year of Powerwall data (uses saved email)
  %(prog)s download-data

  # Run comparison with saved defaults
  %(prog)s default --power-csv download/1689188759506284/power.csv

  # Override a saved default for one run
  %(prog)s default --power-csv power.csv --region-code C
""",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── set-defaults ──────────────────────────────────────────────
    sd = sub.add_parser("set-defaults",
        help="Save default settings so you don't have to type them every time",
        description="Save default values to .octopus_defaults.json. "
                    "These are used automatically unless overridden on the command line.")
    sd.add_argument("--region-code", help="Octopus region code (run list-regions to see options)")
    sd.add_argument("--battery-capacity-kwh", type=float, help="Powerwall usable capacity in kWh (e.g. 13.0 for a single PW2)")
    sd.add_argument("--extra-daily-kwh", type=float, help="Extra daily consumption to model, in kWh (e.g. heat pump)")
    sd.add_argument("--extra-start", help="Start of extra consumption window (HH:MM)")
    sd.add_argument("--extra-end", help="End of extra consumption window (HH:MM)")
    sd.add_argument("--email", help="Tesla account email for download-data")
    sd.add_argument("--out-dir", help="Directory for comparison output CSVs")
    sd.add_argument("--cache-dir", help="Directory for cached Agile tariff CSVs")
    sd.add_argument("--intelligent-offpeak-import", type=float, help="Intelligent off-peak import rate (p/kWh)")
    sd.add_argument("--intelligent-peak-import", type=float, help="Intelligent peak import rate (p/kWh)")
    sd.add_argument("--intelligent-export", type=float, help="Intelligent export rate (p/kWh)")
    sd.add_argument("--intelligent-standing-charge", type=float, help="Intelligent daily standing charge (p/day)")
    sd.add_argument("--intelligent-offpeak-start", help="Intelligent off-peak window start (HH:MM)")
    sd.add_argument("--intelligent-offpeak-end", help="Intelligent off-peak window end (HH:MM)")
    sd.add_argument("--agile-standing-charge", type=float, help="Agile daily standing charge (p/day)")
    sd.add_argument("--agile-flex-hours", type=float, help="Agile flexible charging hours per day")
    sd.add_argument("--agile-flex-max-kw", type=float, help="Agile max flexible charge rate (kW)")
    sd.add_argument("--show", action="store_true", help="Show current saved defaults and exit")

    # ── default ───────────────────────────────────────────────────
    default = sub.add_parser("default",
        help="Quick comparison using sensible defaults",
        description="Run a tariff comparison with minimal configuration. "
                    "Automatically refreshes Powerwall data from Tesla if --email is saved. "
                    "Uses saved defaults from set-defaults where available.")
    default.add_argument("--power-csv", default="download/power.csv",
        help="Path to Powerwall power CSV (default: %(default)s)")
    default.add_argument("--email", default=d("email", None),
        help="Tesla email - if provided, auto-downloads latest data before comparing")
    default.add_argument("--out-dir", default=d("out_dir", "output"),
        help="Directory for output CSVs (default: %(default)s)")
    default.add_argument("--cache-dir", default=d("cache_dir", ".cache/tariffs"),
        help="Directory for cached Agile tariff CSVs (default: %(default)s)")
    default.add_argument("--region-code", default=d("region_code", "M"),
        help="Octopus region code, e.g. M=Yorkshire, C=London (default: %(default)s)")
    default.add_argument("--extra-daily-kwh", type=float, default=d("extra_daily_kwh", 0.0),
        help="Extra daily kWh to model, e.g. for a heat pump (default: %(default)s)")
    default.add_argument("--battery-capacity-kwh", type=float, default=d("battery_capacity_kwh", 13.0),
        help="Powerwall usable capacity in kWh (default: %(default)s)")
    default.add_argument("--start-date",
        help="Only include data from this date onwards (YYYY-MM-DD)")
    default.add_argument("--end-date",
        help="Only include data up to this date (YYYY-MM-DD)")
    default.add_argument("--refresh-tariffs", action="store_true",
        help="Re-download Agile tariff CSVs even if cached")
    default.add_argument("--no-ev-exclusion", action="store_true",
        help="Disable automatic EV charging detection and exclusion")
    default.add_argument("--tariffs",
        help="Comma-separated list of tariffs to compare (default: all). "
             "Options: intelligent,agile,go,flux,cosy,flexible,tracker")
    default.set_defaults(
        intelligent_offpeak_import=d("intelligent_offpeak_import", 7.0),
        intelligent_peak_import=d("intelligent_peak_import", 26.0),
        intelligent_export=d("intelligent_export", 15.0),
        intelligent_standing_charge=d("intelligent_standing_charge", 57.01),
        intelligent_offpeak_start=d("intelligent_offpeak_start", "23:30"),
        intelligent_offpeak_end=d("intelligent_offpeak_end", "05:30"),
        agile_standing_charge=d("agile_standing_charge", 66.26),
        agile_flex_hours=d("agile_flex_hours", 6.0),
        agile_flex_max_kw=d("agile_flex_max_kw", 3.3),
        extra_start=d("extra_start", "17:00"),
        extra_end=d("extra_end", "22:00"),
        go_offpeak_import=d("go_offpeak_import", 8.0),
        go_peak_import=d("go_peak_import", 24.5),
        go_export=d("go_export", 12.0),
        go_standing_charge=d("go_standing_charge", 53.35),
        go_offpeak_start=d("go_offpeak_start", "23:30"),
        go_offpeak_end=d("go_offpeak_end", "05:30"),
        flux_offpeak_import=d("flux_offpeak_import", 9.80),
        flux_day_import=d("flux_day_import", 22.36),
        flux_peak_import=d("flux_peak_import", 33.54),
        flux_offpeak_export=d("flux_offpeak_export", 4.05),
        flux_day_export=d("flux_day_export", 14.40),
        flux_peak_export=d("flux_peak_export", 28.60),
        flux_standing_charge=d("flux_standing_charge", 48.93),
        flux_offpeak_start=d("flux_offpeak_start", "02:00"),
        flux_offpeak_end=d("flux_offpeak_end", "05:00"),
        flux_peak_start=d("flux_peak_start", "16:00"),
        flux_peak_end=d("flux_peak_end", "19:00"),
        cosy_import=d("cosy_import", 12.0),
        cosy_day_import=d("cosy_day_import", 24.50),
        cosy_peak_import=d("cosy_peak_import", 36.75),
        cosy_export=d("cosy_export", 12.0),
        cosy_standing_charge=d("cosy_standing_charge", 53.35),
        flexible_import=d("flexible_import", 24.50),
        flexible_export=d("flexible_export", 12.0),
        flexible_standing_charge=d("flexible_standing_charge", 53.35),
        tracker_standing_charge=d("tracker_standing_charge", 53.35),
        tracker_export=d("tracker_export", 12.0),
    )

    # ── compare ───────────────────────────────────────────────────
    compare = sub.add_parser("compare",
        help="Full comparison with all tariff parameters exposed",
        description="Run a tariff comparison with full control over every rate and window.")
    compare.add_argument("--power-csv", required=True,
        help="Path to Powerwall power CSV (5-min intervals with timestamp and grid_power columns)")
    compare.add_argument("--out-dir", default=d("out_dir", "output"),
        help="Directory for output CSVs (default: %(default)s)")
    compare.add_argument("--cache-dir", default=d("cache_dir", ".cache/tariffs"),
        help="Directory for cached Agile tariff CSVs (default: %(default)s)")
    compare.add_argument("--region-code", default=d("region_code", "M"),
        help="Octopus region code (default: %(default)s)")
    compare.add_argument("--start-date",
        help="Only include data from this date onwards (YYYY-MM-DD)")
    compare.add_argument("--end-date",
        help="Only include data up to this date (YYYY-MM-DD)")
    compare.add_argument("--refresh-tariffs", action="store_true",
        help="Re-download Agile tariff CSVs even if cached")
    compare.add_argument("--battery-capacity-kwh", type=float, default=d("battery_capacity_kwh", 13.0),
        help="Powerwall usable capacity in kWh (default: %(default)s)")
    compare.add_argument("--extra-daily-kwh", type=float, default=d("extra_daily_kwh", 0.0),
        help="Extra daily kWh to model (default: %(default)s)")
    compare.add_argument("--extra-start", default=d("extra_start", "17:00"),
        help="Start of extra consumption window HH:MM (default: %(default)s)")
    compare.add_argument("--extra-end", default=d("extra_end", "22:00"),
        help="End of extra consumption window HH:MM (default: %(default)s)")
    compare.add_argument("--intelligent-offpeak-import", type=float, default=d("intelligent_offpeak_import", 7.0),
        help="Intelligent off-peak import rate in p/kWh (default: %(default)s)")
    compare.add_argument("--intelligent-peak-import", type=float, default=d("intelligent_peak_import", 26.0),
        help="Intelligent peak import rate in p/kWh (default: %(default)s)")
    compare.add_argument("--intelligent-export", type=float, default=d("intelligent_export", 15.0),
        help="Intelligent export rate in p/kWh (default: %(default)s)")
    compare.add_argument("--intelligent-standing-charge", type=float, default=d("intelligent_standing_charge", 57.01),
        help="Intelligent daily standing charge in p/day (default: %(default)s)")
    compare.add_argument("--intelligent-offpeak-start", default=d("intelligent_offpeak_start", "23:30"),
        help="Intelligent off-peak window start HH:MM (default: %(default)s)")
    compare.add_argument("--intelligent-offpeak-end", default=d("intelligent_offpeak_end", "05:30"),
        help="Intelligent off-peak window end HH:MM (default: %(default)s)")
    compare.add_argument("--agile-standing-charge", type=float, default=d("agile_standing_charge", 66.26),
        help="Agile daily standing charge in p/day (default: %(default)s)")
    compare.add_argument("--agile-flex-hours", type=float, default=d("agile_flex_hours", 6.0),
        help="Hours per day of flexible Agile charging (default: %(default)s)")
    compare.add_argument("--agile-flex-max-kw", type=float, default=d("agile_flex_max_kw", 3.3),
        help="Max charge rate during flexible Agile slots in kW (default: %(default)s)")
    compare.add_argument("--no-ev-exclusion", action="store_true",
        help="Disable automatic EV charging detection and exclusion")
    compare.add_argument("--tariffs",
        help="Comma-separated list of tariffs to compare (default: all). "
             "Options: intelligent,agile,go,flux,cosy,flexible,tracker")
    # Go tariff arguments
    compare.add_argument("--go-offpeak-import", type=float, default=d("go_offpeak_import", 8.0),
        help="Go off-peak import rate in p/kWh (default: %(default)s)")
    compare.add_argument("--go-peak-import", type=float, default=d("go_peak_import", 24.5),
        help="Go peak import rate in p/kWh (default: %(default)s)")
    compare.add_argument("--go-export", type=float, default=d("go_export", 12.0),
        help="Go export rate in p/kWh (default: %(default)s)")
    compare.add_argument("--go-standing-charge", type=float, default=d("go_standing_charge", 53.35),
        help="Go daily standing charge in p/day (default: %(default)s)")
    compare.add_argument("--go-offpeak-start", default=d("go_offpeak_start", "23:30"),
        help="Go off-peak window start HH:MM (default: %(default)s)")
    compare.add_argument("--go-offpeak-end", default=d("go_offpeak_end", "05:30"),
        help="Go off-peak window end HH:MM (default: %(default)s)")
    # Flux tariff arguments
    compare.add_argument("--flux-offpeak-import", type=float, default=d("flux_offpeak_import", 9.80),
        help="Flux off-peak import rate in p/kWh (default: %(default)s)")
    compare.add_argument("--flux-day-import", type=float, default=d("flux_day_import", 22.36),
        help="Flux day import rate in p/kWh (default: %(default)s)")
    compare.add_argument("--flux-peak-import", type=float, default=d("flux_peak_import", 33.54),
        help="Flux peak import rate in p/kWh (default: %(default)s)")
    compare.add_argument("--flux-offpeak-export", type=float, default=d("flux_offpeak_export", 4.05),
        help="Flux off-peak export rate in p/kWh (default: %(default)s)")
    compare.add_argument("--flux-day-export", type=float, default=d("flux_day_export", 14.40),
        help="Flux day export rate in p/kWh (default: %(default)s)")
    compare.add_argument("--flux-peak-export", type=float, default=d("flux_peak_export", 28.60),
        help="Flux peak export rate in p/kWh (default: %(default)s)")
    compare.add_argument("--flux-standing-charge", type=float, default=d("flux_standing_charge", 48.93),
        help="Flux daily standing charge in p/day (default: %(default)s)")
    compare.add_argument("--flux-offpeak-start", default=d("flux_offpeak_start", "02:00"),
        help="Flux off-peak window start HH:MM (default: %(default)s)")
    compare.add_argument("--flux-offpeak-end", default=d("flux_offpeak_end", "05:00"),
        help="Flux off-peak window end HH:MM (default: %(default)s)")
    compare.add_argument("--flux-peak-start", default=d("flux_peak_start", "16:00"),
        help="Flux peak window start HH:MM (default: %(default)s)")
    compare.add_argument("--flux-peak-end", default=d("flux_peak_end", "19:00"),
        help="Flux peak window end HH:MM (default: %(default)s)")
    # Cosy tariff arguments
    compare.add_argument("--cosy-import", type=float, default=d("cosy_import", 12.0),
        help="Cosy cheap window import rate in p/kWh (default: %(default)s)")
    compare.add_argument("--cosy-day-import", type=float, default=d("cosy_day_import", 24.50),
        help="Cosy standard day import rate in p/kWh (default: %(default)s)")
    compare.add_argument("--cosy-peak-import", type=float, default=d("cosy_peak_import", 36.75),
        help="Cosy peak import rate in p/kWh (default: %(default)s)")
    compare.add_argument("--cosy-export", type=float, default=d("cosy_export", 12.0),
        help="Cosy export rate in p/kWh (default: %(default)s)")
    compare.add_argument("--cosy-standing-charge", type=float, default=d("cosy_standing_charge", 53.35),
        help="Cosy daily standing charge in p/day (default: %(default)s)")
    # Flexible (SVR) tariff arguments
    compare.add_argument("--flexible-import", type=float, default=d("flexible_import", 24.50),
        help="Flexible (SVR) flat import rate in p/kWh (default: %(default)s)")
    compare.add_argument("--flexible-export", type=float, default=d("flexible_export", 12.0),
        help="Flexible export rate in p/kWh (default: %(default)s)")
    compare.add_argument("--flexible-standing-charge", type=float, default=d("flexible_standing_charge", 53.35),
        help="Flexible daily standing charge in p/day (default: %(default)s)")
    # Tracker tariff arguments
    compare.add_argument("--tracker-standing-charge", type=float, default=d("tracker_standing_charge", 53.35),
        help="Tracker daily standing charge in p/day (default: %(default)s)")
    compare.add_argument("--tracker-export", type=float, default=d("tracker_export", 12.0),
        help="Tracker export rate in p/kWh (default: %(default)s)")

    # ── download-data ─────────────────────────────────────────────
    download = sub.add_parser("download-data",
        help="Download a year of 5-minute Powerwall data from Tesla",
        description="Logs in to Tesla, downloads power data day-by-day in 5-minute intervals, "
                    "merges into a single CSV, and cleans up per-day files.")
    download.add_argument("--email", default=d("email", None),
        help="Tesla account email address" + (f" (default: {d('email', None)})" if d("email", None) else ""))

    # ── list-regions ──────────────────────────────────────────────
    regions = sub.add_parser("list-regions",
        help="Show supported Octopus region codes and names")
    regions.add_argument("--json", action="store_true",
        help="Output as JSON instead of plain text")

    # ── refresh-tariffs ───────────────────────────────────────────
    refresh = sub.add_parser("refresh-tariffs",
        help="Re-download Agile tariff CSVs into cache",
        description="Force re-download of Agile import and export tariff CSVs for a region.")
    refresh.add_argument("--region-code", default=d("region_code", "M"),
        help="Octopus region code (default: %(default)s)")
    refresh.add_argument("--cache-dir", default=d("cache_dir", ".cache/tariffs"),
        help="Directory for cached tariff CSVs (default: %(default)s)")

    # ── full-refresh ──────────────────────────────────────────────
    full = sub.add_parser("full-refresh",
        help="Download latest Powerwall data, refresh tariffs, and run comparison in one go",
        description="All-in-one: downloads latest Powerwall data from Tesla, "
                    "refreshes Agile tariff CSVs, and runs the tariff comparison.")
    full.add_argument("--email", default=d("email", None),
        help="Tesla account email address")
    full.add_argument("--out-dir", default=d("out_dir", "output"),
        help="Directory for output CSVs (default: %(default)s)")
    full.add_argument("--cache-dir", default=d("cache_dir", ".cache/tariffs"),
        help="Directory for cached Agile tariff CSVs (default: %(default)s)")
    full.add_argument("--region-code", default=d("region_code", "M"),
        help="Octopus region code (default: %(default)s)")
    full.add_argument("--battery-capacity-kwh", type=float, default=d("battery_capacity_kwh", 13.0),
        help="Powerwall usable capacity in kWh (default: %(default)s)")
    full.add_argument("--no-ev-exclusion", action="store_true",
        help="Disable automatic EV charging detection and exclusion")
    full.set_defaults(
        extra_daily_kwh=d("extra_daily_kwh", 0.0),
        extra_start=d("extra_start", "17:00"),
        extra_end=d("extra_end", "22:00"),
        intelligent_offpeak_import=d("intelligent_offpeak_import", 7.0),
        intelligent_peak_import=d("intelligent_peak_import", 26.0),
        intelligent_export=d("intelligent_export", 15.0),
        intelligent_standing_charge=d("intelligent_standing_charge", 57.01),
        intelligent_offpeak_start=d("intelligent_offpeak_start", "23:30"),
        intelligent_offpeak_end=d("intelligent_offpeak_end", "05:30"),
        agile_standing_charge=d("agile_standing_charge", 66.26),
        agile_flex_hours=d("agile_flex_hours", 6.0),
        agile_flex_max_kw=d("agile_flex_max_kw", 3.3),
        go_offpeak_import=d("go_offpeak_import", 8.0),
        go_peak_import=d("go_peak_import", 24.5),
        go_export=d("go_export", 12.0),
        go_standing_charge=d("go_standing_charge", 53.35),
        go_offpeak_start=d("go_offpeak_start", "23:30"),
        go_offpeak_end=d("go_offpeak_end", "05:30"),
        flux_offpeak_import=d("flux_offpeak_import", 9.80),
        flux_day_import=d("flux_day_import", 22.36),
        flux_peak_import=d("flux_peak_import", 33.54),
        flux_offpeak_export=d("flux_offpeak_export", 4.05),
        flux_day_export=d("flux_day_export", 14.40),
        flux_peak_export=d("flux_peak_export", 28.60),
        flux_standing_charge=d("flux_standing_charge", 48.93),
        flux_offpeak_start=d("flux_offpeak_start", "02:00"),
        flux_offpeak_end=d("flux_offpeak_end", "05:00"),
        flux_peak_start=d("flux_peak_start", "16:00"),
        flux_peak_end=d("flux_peak_end", "19:00"),
        cosy_import=d("cosy_import", 12.0),
        cosy_day_import=d("cosy_day_import", 24.50),
        cosy_peak_import=d("cosy_peak_import", 36.75),
        cosy_export=d("cosy_export", 12.0),
        cosy_standing_charge=d("cosy_standing_charge", 53.35),
        flexible_import=d("flexible_import", 24.50),
        flexible_export=d("flexible_export", 12.0),
        flexible_standing_charge=d("flexible_standing_charge", 53.35),
        tracker_standing_charge=d("tracker_standing_charge", 53.35),
        tracker_export=d("tracker_export", 12.0),
        start_date=None, end_date=None, refresh_tariffs=True,
        tariffs=None,
    )

    # ── model ─────────────────────────────────────────────────────
    model = sub.add_parser("model",
        help="Model a scenario with changed energy usage (e.g. new EV, hot tub)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""Model how a change in energy usage would affect your tariff costs.

Takes your real Powerwall data and applies adjustments to simulate scenarios
like buying an EV, adding a hot tub, or installing extra solar panels.

Adjustments are applied as extra daily kWh consumed (positive) or reduced (negative)
during a specified time window.""",
        epilog="""examples:
  # Model adding an EV that charges 7.5 kWh/day overnight
  %(prog)s model --power-csv power.csv --label "New EV" --adjust-kwh 7.5 --window 00:30-05:30

  # Model a hot tub using 5 kWh/day in the evening
  %(prog)s model --power-csv power.csv --label "Hot tub" --adjust-kwh 5 --window 17:00-22:00

  # Model reduced usage from better insulation (-3 kWh/day spread across peak)
  %(prog)s model --power-csv power.csv --label "Insulation" --adjust-kwh -3 --window 06:00-22:00

  # Stack multiple scenarios
  %(prog)s model --power-csv power.csv \\
      --label "EV" --adjust-kwh 7.5 --window 00:30-05:30 \\
      --label "Hot tub" --adjust-kwh 5 --window 17:00-22:00
""")
    model.add_argument("--power-csv", required=True,
        help="Path to Powerwall power CSV")
    model.add_argument("--label", action="append", dest="labels", required=True,
        help="Name for this scenario adjustment (repeat for multiple)")
    model.add_argument("--adjust-kwh", action="append", dest="adjust_kwhs", type=float, required=True,
        help="Daily kWh change: positive=more usage, negative=less (repeat for multiple)")
    model.add_argument("--window", action="append", dest="windows", required=True,
        help="Time window for the adjustment as HH:MM-HH:MM (repeat for multiple)")
    model.add_argument("--out-dir", default=d("out_dir", "output"),
        help="Directory for output CSVs (default: %(default)s)")
    model.add_argument("--cache-dir", default=d("cache_dir", ".cache/tariffs"),
        help="Directory for cached Agile tariff CSVs (default: %(default)s)")
    model.add_argument("--region-code", default=d("region_code", "M"),
        help="Octopus region code (default: %(default)s)")
    model.add_argument("--battery-capacity-kwh", type=float, default=d("battery_capacity_kwh", 13.0),
        help="Powerwall usable capacity in kWh (default: %(default)s)")
    model.add_argument("--no-ev-exclusion", action="store_true",
        help="Disable automatic EV charging detection and exclusion")
    model.set_defaults(
        extra_daily_kwh=0.0,
        extra_start="17:00", extra_end="22:00",
        intelligent_offpeak_import=d("intelligent_offpeak_import", 7.0),
        intelligent_peak_import=d("intelligent_peak_import", 26.0),
        intelligent_export=d("intelligent_export", 15.0),
        intelligent_standing_charge=d("intelligent_standing_charge", 57.01),
        intelligent_offpeak_start=d("intelligent_offpeak_start", "23:30"),
        intelligent_offpeak_end=d("intelligent_offpeak_end", "05:30"),
        agile_standing_charge=d("agile_standing_charge", 66.26),
        agile_flex_hours=d("agile_flex_hours", 6.0),
        agile_flex_max_kw=d("agile_flex_max_kw", 3.3),
        go_offpeak_import=d("go_offpeak_import", 8.0),
        go_peak_import=d("go_peak_import", 24.5),
        go_export=d("go_export", 12.0),
        go_standing_charge=d("go_standing_charge", 53.35),
        go_offpeak_start=d("go_offpeak_start", "23:30"),
        go_offpeak_end=d("go_offpeak_end", "05:30"),
        flux_offpeak_import=d("flux_offpeak_import", 9.80),
        flux_day_import=d("flux_day_import", 22.36),
        flux_peak_import=d("flux_peak_import", 33.54),
        flux_offpeak_export=d("flux_offpeak_export", 4.05),
        flux_day_export=d("flux_day_export", 14.40),
        flux_peak_export=d("flux_peak_export", 28.60),
        flux_standing_charge=d("flux_standing_charge", 48.93),
        flux_offpeak_start=d("flux_offpeak_start", "02:00"),
        flux_offpeak_end=d("flux_offpeak_end", "05:00"),
        flux_peak_start=d("flux_peak_start", "16:00"),
        flux_peak_end=d("flux_peak_end", "19:00"),
        cosy_import=d("cosy_import", 12.0),
        cosy_day_import=d("cosy_day_import", 24.50),
        cosy_peak_import=d("cosy_peak_import", 36.75),
        cosy_export=d("cosy_export", 12.0),
        cosy_standing_charge=d("cosy_standing_charge", 53.35),
        flexible_import=d("flexible_import", 24.50),
        flexible_export=d("flexible_export", 12.0),
        flexible_standing_charge=d("flexible_standing_charge", 53.35),
        tracker_standing_charge=d("tracker_standing_charge", 53.35),
        tracker_export=d("tracker_export", 12.0),
        start_date=None, end_date=None, refresh_tariffs=False,
        tariffs=None,
    )

    # ── optimise-charging ─────────────────────────────────────────
    opt = sub.add_parser("optimise-charging",
        help="Show savings from moving battery charging to cheapest slots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""Analyse your actual battery charging patterns and calculate how much
you could save by shifting charging to the cheapest time slots on each tariff.

This looks at when your Powerwall actually charged from the grid, then simulates
moving that same energy to the cheapest available half-hour slots each day.

For time-of-use tariffs (Intelligent, Go, Flux, Cosy), this shows the benefit
of ensuring all charging happens in off-peak windows.

For variable tariffs (Agile, Tracker), this shows the benefit of smart scheduling
to hit the cheapest half-hours each day.""",
        epilog="""examples:
  # See savings across all tariffs
  %(prog)s optimise-charging --power-csv download/power.csv

  # Just check Agile and Tracker
  %(prog)s optimise-charging --power-csv download/power.csv --tariffs agile,tracker

  # With a higher charge rate (e.g. Powerwall 3 at 11.5 kW)
  %(prog)s optimise-charging --power-csv download/power.csv --battery-max-charge-kw 11.5
""")
    opt.add_argument("--power-csv", default="download/power.csv",
        help="Path to Powerwall power CSV (default: %(default)s)")
    opt.add_argument("--out-dir", default=d("out_dir", "output"),
        help="Directory for output CSVs (default: %(default)s)")
    opt.add_argument("--cache-dir", default=d("cache_dir", ".cache/tariffs"),
        help="Directory for cached tariff CSVs (default: %(default)s)")
    opt.add_argument("--region-code", default=d("region_code", "M"),
        help="Octopus region code (default: %(default)s)")
    opt.add_argument("--start-date",
        help="Only include data from this date onwards (YYYY-MM-DD)")
    opt.add_argument("--end-date",
        help="Only include data up to this date (YYYY-MM-DD)")
    opt.add_argument("--battery-capacity-kwh", type=float, default=d("battery_capacity_kwh", 13.0),
        help="Powerwall usable capacity in kWh (default: %(default)s)")
    opt.add_argument("--battery-max-charge-kw", type=float, default=5.0,
        help="Maximum battery charge rate in kW (default: %(default)s, Powerwall 2)")
    opt.add_argument("--refresh-tariffs", action="store_true",
        help="Re-download tariff data even if cached")
    opt.add_argument("--no-ev-exclusion", action="store_true",
        help="Disable automatic EV charging detection and exclusion")
    opt.add_argument("--tariffs",
        help="Comma-separated list of tariffs to analyse (default: all). "
             "Options: intelligent,agile,go,flux,cosy,flexible,tracker")
    opt.set_defaults(
        extra_daily_kwh=d("extra_daily_kwh", 0.0),
        extra_start=d("extra_start", "17:00"),
        extra_end=d("extra_end", "22:00"),
        intelligent_offpeak_import=d("intelligent_offpeak_import", 7.0),
        intelligent_peak_import=d("intelligent_peak_import", 26.0),
        intelligent_export=d("intelligent_export", 15.0),
        intelligent_standing_charge=d("intelligent_standing_charge", 57.01),
        intelligent_offpeak_start=d("intelligent_offpeak_start", "23:30"),
        intelligent_offpeak_end=d("intelligent_offpeak_end", "05:30"),
        agile_standing_charge=d("agile_standing_charge", 66.26),
        agile_flex_hours=d("agile_flex_hours", 6.0),
        agile_flex_max_kw=d("agile_flex_max_kw", 3.3),
        go_offpeak_import=d("go_offpeak_import", 8.0),
        go_peak_import=d("go_peak_import", 24.5),
        go_export=d("go_export", 12.0),
        go_standing_charge=d("go_standing_charge", 53.35),
        go_offpeak_start=d("go_offpeak_start", "23:30"),
        go_offpeak_end=d("go_offpeak_end", "05:30"),
        flux_offpeak_import=d("flux_offpeak_import", 9.80),
        flux_day_import=d("flux_day_import", 22.36),
        flux_peak_import=d("flux_peak_import", 33.54),
        flux_offpeak_export=d("flux_offpeak_export", 4.05),
        flux_day_export=d("flux_day_export", 14.40),
        flux_peak_export=d("flux_peak_export", 28.60),
        flux_standing_charge=d("flux_standing_charge", 48.93),
        flux_offpeak_start=d("flux_offpeak_start", "02:00"),
        flux_offpeak_end=d("flux_offpeak_end", "05:00"),
        flux_peak_start=d("flux_peak_start", "16:00"),
        flux_peak_end=d("flux_peak_end", "19:00"),
        cosy_import=d("cosy_import", 12.0),
        cosy_day_import=d("cosy_day_import", 24.50),
        cosy_peak_import=d("cosy_peak_import", 36.75),
        cosy_export=d("cosy_export", 12.0),
        cosy_standing_charge=d("cosy_standing_charge", 53.35),
        flexible_import=d("flexible_import", 24.50),
        flexible_export=d("flexible_export", 12.0),
        flexible_standing_charge=d("flexible_standing_charge", 53.35),
        tracker_standing_charge=d("tracker_standing_charge", 53.35),
        tracker_export=d("tracker_export", 12.0),
    )

    # ── best-tariff ───────────────────────────────────────────────
    best = sub.add_parser("best-tariff",
        help="Full simulation: find the absolute best tariff with optimal battery scheduling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""Run a comprehensive simulation to find the best tariff for your home.

Unlike the basic 'compare' command which just applies rates to your actual usage,
this simulates OPTIMAL battery behaviour for each tariff:

  - Charges the battery from grid during the cheapest import slots
  - Discharges to avoid importing during expensive periods
  - Exports stored energy during peak export windows (e.g. Flux 16:00-19:00)
  - Accounts for solar generation, home load, and battery round-trip losses
  - Respects battery capacity and charge/discharge rate limits

This shows the true potential of each tariff if you had perfect battery scheduling
(e.g. via Intelligent Octopus, Flux automation, or a smart controller).""",
        epilog="""examples:
  # Find the best tariff with optimal battery use
  %(prog)s best-tariff --power-csv download/power.csv

  # Compare just the smart tariffs
  %(prog)s best-tariff --power-csv download/power.csv --tariffs agile,flux,intelligent

  # With Powerwall 3 specs
  %(prog)s best-tariff --power-csv download/power.csv \\
      --battery-max-charge-kw 11.5 --battery-max-discharge-kw 11.5

  # Account for battery degradation (85% efficiency)
  %(prog)s best-tariff --power-csv download/power.csv --battery-efficiency 0.85
""")
    best.add_argument("--power-csv", default="download/power.csv",
        help="Path to Powerwall power CSV (default: %(default)s)")
    best.add_argument("--out-dir", default=d("out_dir", "output"),
        help="Directory for output CSVs (default: %(default)s)")
    best.add_argument("--cache-dir", default=d("cache_dir", ".cache/tariffs"),
        help="Directory for cached tariff CSVs (default: %(default)s)")
    best.add_argument("--region-code", default=d("region_code", "M"),
        help="Octopus region code (default: %(default)s)")
    best.add_argument("--start-date",
        help="Only include data from this date onwards (YYYY-MM-DD)")
    best.add_argument("--end-date",
        help="Only include data up to this date (YYYY-MM-DD)")
    best.add_argument("--battery-capacity-kwh", type=float, default=d("battery_capacity_kwh", 13.0),
        help="Powerwall usable capacity in kWh (default: %(default)s)")
    best.add_argument("--battery-max-charge-kw", type=float, default=5.0,
        help="Maximum battery charge rate in kW (default: %(default)s)")
    best.add_argument("--battery-max-discharge-kw", type=float, default=5.0,
        help="Maximum battery discharge rate in kW (default: %(default)s)")
    best.add_argument("--battery-efficiency", type=float, default=0.90,
        help="Battery round-trip efficiency 0-1 (default: %(default)s)")
    best.add_argument("--refresh-tariffs", action="store_true",
        help="Re-download tariff data even if cached")
    best.add_argument("--no-ev-exclusion", action="store_true",
        help="Disable automatic EV charging detection and exclusion")
    best.add_argument("--tariffs",
        help="Comma-separated list of tariffs to simulate (default: all). "
             "Options: intelligent,agile,go,flux,cosy,flexible,tracker")
    best.set_defaults(
        extra_daily_kwh=d("extra_daily_kwh", 0.0),
        extra_start=d("extra_start", "17:00"),
        extra_end=d("extra_end", "22:00"),
        intelligent_offpeak_import=d("intelligent_offpeak_import", 7.0),
        intelligent_peak_import=d("intelligent_peak_import", 26.0),
        intelligent_export=d("intelligent_export", 15.0),
        intelligent_standing_charge=d("intelligent_standing_charge", 57.01),
        intelligent_offpeak_start=d("intelligent_offpeak_start", "23:30"),
        intelligent_offpeak_end=d("intelligent_offpeak_end", "05:30"),
        agile_standing_charge=d("agile_standing_charge", 66.26),
        agile_flex_hours=d("agile_flex_hours", 6.0),
        agile_flex_max_kw=d("agile_flex_max_kw", 3.3),
        go_offpeak_import=d("go_offpeak_import", 8.0),
        go_peak_import=d("go_peak_import", 24.5),
        go_export=d("go_export", 12.0),
        go_standing_charge=d("go_standing_charge", 53.35),
        go_offpeak_start=d("go_offpeak_start", "23:30"),
        go_offpeak_end=d("go_offpeak_end", "05:30"),
        flux_offpeak_import=d("flux_offpeak_import", 9.80),
        flux_day_import=d("flux_day_import", 22.36),
        flux_peak_import=d("flux_peak_import", 33.54),
        flux_offpeak_export=d("flux_offpeak_export", 4.05),
        flux_day_export=d("flux_day_export", 14.40),
        flux_peak_export=d("flux_peak_export", 28.60),
        flux_standing_charge=d("flux_standing_charge", 48.93),
        flux_offpeak_start=d("flux_offpeak_start", "02:00"),
        flux_offpeak_end=d("flux_offpeak_end", "05:00"),
        flux_peak_start=d("flux_peak_start", "16:00"),
        flux_peak_end=d("flux_peak_end", "19:00"),
        cosy_import=d("cosy_import", 12.0),
        cosy_day_import=d("cosy_day_import", 24.50),
        cosy_peak_import=d("cosy_peak_import", 36.75),
        cosy_export=d("cosy_export", 12.0),
        cosy_standing_charge=d("cosy_standing_charge", 53.35),
        flexible_import=d("flexible_import", 24.50),
        flexible_export=d("flexible_export", 12.0),
        flexible_standing_charge=d("flexible_standing_charge", 53.35),
        tracker_standing_charge=d("tracker_standing_charge", 53.35),
        tracker_export=d("tracker_export", 12.0),
    )

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "set-defaults":
        return run_set_defaults(args)
    if args.command == "download-data":
        if not args.email:
            print("Error: --email is required (or save it with: set-defaults --email you@example.com)")
            return 1
        return run_download_data(args)
    if args.command == "full-refresh":
        return run_full_refresh(args)
    if args.command == "model":
        return run_model(args)
    if args.command in {"default", "compare"}:
        # Auto-download latest data if email is available and using default power-csv path
        if args.command == "default" and getattr(args, "email", None):
            power_csv_path = Path(args.power_csv)
            if str(power_csv_path) == "download/power.csv":
                print("Refreshing Powerwall data from Tesla...\n")
                rc = run_download_data(args)
                if rc != 0:
                    return rc
                print()
            elif not power_csv_path.exists():
                print(f"Error: {args.power_csv} not found.")
                print("Run with --email to download data, or specify a valid --power-csv path.")
                return 1
        elif args.command == "default" and not Path(args.power_csv).exists():
            print(f"Error: {args.power_csv} not found.")
            print("Either:")
            print("  1. Save your email: set-defaults --email you@example.com")
            print("     Then run 'default' again to auto-download from Tesla")
            print("  2. Specify a CSV: default --power-csv /path/to/power.csv")
            return 1
        return run_compare(args)
    if args.command == "list-regions":
        if args.json:
            print(json.dumps(REGION_CODE_TO_NAME, indent=2))
        else:
            for code, name in REGION_CODE_TO_NAME.items():
                print(f"{code}: {name}")
        return 0
    if args.command == "refresh-tariffs":
        in_path, out_path = download_region_tariffs(args.region_code, Path(args.cache_dir), force=True)
        print(f"Downloaded:\n- {in_path}\n- {out_path}")
        return 0
    if args.command == "optimise-charging":
        return run_optimise_charging(args)
    if args.command == "best-tariff":
        return run_best_tariff(args)
    return 0


def run_set_defaults(args) -> int:
    """Save or show user defaults."""
    if args.show:
        saved = load_defaults()
        if not saved:
            print("No defaults saved yet. Use set-defaults with options to save them.")
            print("Example: set-defaults --region-code M --battery-capacity-kwh 13 --email you@example.com")
        else:
            print("Saved defaults:")
            for key, value in sorted(saved.items()):
                print(f"  {key}: {value}")
        return 0

    # Collect any provided values
    saved = load_defaults()
    updated = False
    for key in SAVEABLE_DEFAULTS:
        val = getattr(args, key, None)
        if val is not None:
            saved[key] = val
            updated = True

    if not updated:
        print("No values provided. Use --show to see current defaults, or pass options to save.")
        print("Example: set-defaults --region-code M --battery-capacity-kwh 13 --email you@example.com")
        return 0

    save_defaults(saved)
    print("Defaults saved to .octopus_defaults.json:")
    for key, value in sorted(saved.items()):
        print(f"  {key}: {value}")
    return 0


def run_full_refresh(args) -> int:
    """Download latest Powerwall data, refresh tariffs, and run comparison."""
    if not args.email:
        print("Error: --email is required (or save it with: set-defaults --email you@example.com)")
        return 1

    # Step 1: Download Powerwall data
    print("=" * 60)
    print("STEP 1: Downloading Powerwall data from Tesla")
    print("=" * 60)
    rc = run_download_data(args)
    if rc != 0:
        return rc

    # Find the downloaded power CSV
    download_dir = Path("download")
    power_csv = download_dir / "power.csv"
    if not power_csv.exists():
        print("Error: No power.csv found after download")
        return 1

    # Step 2: Refresh tariffs
    print()
    print("=" * 60)
    print("STEP 2: Refreshing Agile tariff data")
    print("=" * 60)
    in_path, out_path = download_region_tariffs(args.region_code, Path(args.cache_dir), force=True)
    print(f"Downloaded:\n- {in_path}\n- {out_path}")

    # Step 3: Run comparison
    print()
    print("=" * 60)
    print("STEP 3: Running tariff comparison")
    print("=" * 60)
    args.power_csv = str(power_csv)
    return run_compare(args)


def run_model(args) -> int:
    """Run scenario modelling with adjusted energy usage."""
    labels = args.labels
    adjust_kwhs = args.adjust_kwhs
    windows = args.windows

    if not (len(labels) == len(adjust_kwhs) == len(windows)):
        print("Error: --label, --adjust-kwh, and --window must be provided the same number of times")
        return 1

    # Validate windows
    for w in windows:
        if "-" not in w:
            print(f"Error: window '{w}' must be in HH:MM-HH:MM format")
            return 1
        parts = w.split("-")
        if len(parts) != 2:
            print(f"Error: window '{w}' must be in HH:MM-HH:MM format")
            return 1
        for p in parts:
            try:
                parse_hhmm(p)
            except ValueError:
                print(f"Error: invalid time '{p}' in window '{w}'")
                return 1

    # Run baseline comparison first
    print("=" * 60)
    print("BASELINE: Current usage")
    print("=" * 60)
    rc = run_compare(args)
    if rc != 0:
        return rc

    # Read baseline results
    out_dir = Path(args.out_dir)
    baseline_df = pd.read_csv(out_dir / "summary.csv")
    baseline_rows = baseline_df.to_dict(orient="records")

    # Run each scenario by stacking adjustments
    total_extra_kwh = 0.0
    scenario_descriptions = []

    for i, (label, kwh, window) in enumerate(zip(labels, adjust_kwhs, windows, strict=True)):
        total_extra_kwh += kwh
        start_hhmm, end_hhmm = window.split("-")
        scenario_descriptions.append(f"{label}: {'+' if kwh >= 0 else ''}{kwh} kWh/day ({window})")

        print()
        print("=" * 60)
        scenario_name = " + ".join(
            name for name in labels[:i+1]
        )
        print(f"SCENARIO: {scenario_name}")
        for desc in scenario_descriptions:
            print(f"  {desc}")
        print(f"  Total adjustment: {'+' if total_extra_kwh >= 0 else ''}{total_extra_kwh} kWh/day")
        print("=" * 60)

        # Override extra_daily_kwh and window for this scenario
        args.extra_daily_kwh = total_extra_kwh
        args.extra_start = start_hhmm
        args.extra_end = end_hhmm

        # Use a separate output dir for each scenario
        scenario_dir = out_dir / f"scenario_{i+1}"
        args.out_dir = str(scenario_dir)
        rc = run_compare(args)
        if rc != 0:
            return rc

    # Print combined summary
    print()
    print("=" * 60)
    print("SUMMARY: All scenarios compared")
    print("=" * 60)

    all_rows = []
    # Baseline
    for row in baseline_rows:
        row["scenario"] = "baseline"
        all_rows.append(row)

    # Each scenario
    cumulative_labels = []
    for i, label in enumerate(labels):
        cumulative_labels.append(label)
        scenario_dir = out_dir / f"scenario_{i+1}"
        scenario_df = pd.read_csv(scenario_dir / "summary.csv")
        for row in scenario_df.to_dict(orient="records"):
            row["scenario"] = " + ".join(cumulative_labels)
            all_rows.append(row)

    headers = ["scenario", "tariff", "import_cost_gbp", "export_revenue_gbp", "standing_charge_gbp", "net_cost_gbp"]
    widths = {h: max(len(h), max(len(str(r.get(h, ""))) for r in all_rows)) for h in headers}
    print("  ".join(h.ljust(widths[h]) for h in headers))
    print("  ".join("-" * widths[h] for h in headers))
    for row in all_rows:
        print("  ".join(str(row.get(h, "")).ljust(widths[h]) for h in headers))

    # Save combined summary
    combined_df = pd.DataFrame(all_rows)
    write_csv(combined_df, out_dir / "model_summary.csv")
    print(f"\nModel summary written to: {out_dir / 'model_summary.csv'}")

    # Restore out_dir
    args.out_dir = str(out_dir)
    return 0


def run_optimise_charging(args) -> int:
    """Analyse battery charging patterns and show savings from optimal scheduling."""
    scenario = ScenarioConfig(
        battery_capacity_kwh=args.battery_capacity_kwh,
        extra_daily_kwh=getattr(args, "extra_daily_kwh", 0.0),
        extra_start=getattr(args, "extra_start", "17:00"),
        extra_end=getattr(args, "extra_end", "22:00"),
        ev_exclusion_enabled=not args.no_ev_exclusion,
    )

    power_csv = Path(args.power_csv)
    if not power_csv.exists():
        print(f"Error: {args.power_csv} not found.")
        return 1

    hh = load_power_csv(power_csv, scenario)
    hh = trim_date_range(hh, args.start_date, args.end_date)
    if hh.empty:
        raise RuntimeError("No power data left after applying date filter")

    total_battery_kwh = hh["battery_charge_from_grid_kwh"].sum()
    if total_battery_kwh < 0.1:
        print("No significant battery charging from grid detected in this data.")
        print("This could mean:")
        print("  - Your battery charges primarily from solar")
        print("  - The power CSV doesn't include battery_power column")
        return 0

    num_days = hh["date"].nunique()
    print(f"Battery charging analysis: {total_battery_kwh:.1f} kWh from grid over {num_days} days")
    print(f"Average: {total_battery_kwh / num_days:.1f} kWh/day")
    print(f"Max charge rate assumed: {args.battery_max_charge_kw} kW")
    print()

    # Determine which tariffs to analyse
    tariffs_to_run = getattr(args, "tariffs", None)
    all_tariffs = {"intelligent", "agile", "go", "flux", "cosy", "flexible", "tracker"}
    if tariffs_to_run:
        selected = {t.strip().lower() for t in tariffs_to_run.split(",")}
        invalid = selected - all_tariffs
        if invalid:
            print(f"Error: unknown tariff(s): {', '.join(sorted(invalid))}")
            return 1
    else:
        selected = all_tariffs

    # Build tariff configs
    tariff_configs = {}
    if "intelligent" in selected:
        tariff_configs["intelligent"] = IntelligentTariff(
            offpeak_import_p_per_kwh=args.intelligent_offpeak_import,
            peak_import_p_per_kwh=args.intelligent_peak_import,
            offpeak_start=args.intelligent_offpeak_start,
            offpeak_end=args.intelligent_offpeak_end,
        )
    if "go" in selected:
        tariff_configs["go"] = GoTariff(
            offpeak_import_p_per_kwh=getattr(args, "go_offpeak_import", 8.0),
            peak_import_p_per_kwh=getattr(args, "go_peak_import", 24.5),
            offpeak_start=getattr(args, "go_offpeak_start", "23:30"),
            offpeak_end=getattr(args, "go_offpeak_end", "05:30"),
        )
    if "flux" in selected:
        tariff_configs["flux"] = FluxTariff(
            offpeak_import_p_per_kwh=getattr(args, "flux_offpeak_import", 9.80),
            day_import_p_per_kwh=getattr(args, "flux_day_import", 22.36),
            peak_import_p_per_kwh=getattr(args, "flux_peak_import", 33.54),
            offpeak_start=getattr(args, "flux_offpeak_start", "02:00"),
            offpeak_end=getattr(args, "flux_offpeak_end", "05:00"),
            peak_start=getattr(args, "flux_peak_start", "16:00"),
            peak_end=getattr(args, "flux_peak_end", "19:00"),
        )
    if "cosy" in selected:
        tariff_configs["cosy"] = CosyTariff(
            cosy_import_p_per_kwh=getattr(args, "cosy_import", 12.0),
            day_import_p_per_kwh=getattr(args, "cosy_day_import", 24.50),
            peak_import_p_per_kwh=getattr(args, "cosy_peak_import", 36.75),
        )
    if "flexible" in selected:
        tariff_configs["flexible"] = FlexibleTariff(
            import_p_per_kwh=getattr(args, "flexible_import", 24.50),
        )

    # Fetch variable rate data if needed
    agile_rates = None
    tracker_rates = None
    if "agile" in selected:
        try:
            in_path, _ = download_region_tariffs(args.region_code, Path(args.cache_dir), force=args.refresh_tariffs)
            agile_rates = read_agile_csv(in_path)
            tariff_configs["agile"] = AgileConfig()  # placeholder — rates come from agile_rates
        except (FileNotFoundError, requests.RequestException) as e:
            print(f"  Warning: could not fetch Agile rates: {e}")
    if "tracker" in selected:
        tracker_rates = download_tracker_rates(args.region_code, Path(args.cache_dir), force=args.refresh_tariffs)
        if tracker_rates is not None and not tracker_rates.empty:
            tariff_configs["tracker"] = TrackerTariff()  # placeholder — rates come from tracker_rates
        else:
            print("  Warning: could not fetch Tracker rates, skipping")

    # Run optimisation for each tariff
    summaries = []
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for tariff_name, tariff_cfg in tariff_configs.items():
        s, daily = calculate_optimised_charging(
            hh, tariff_name, tariff_cfg, scenario,
            agile_rates=agile_rates,
            tracker_rates=tracker_rates,
            battery_max_charge_kw=args.battery_max_charge_kw,
        )
        if s is not None:
            summaries.append(s)
            write_csv(daily, out_dir / f"optimised_charging_{tariff_name}.csv")

    if not summaries:
        print("No tariff data available for optimisation analysis.")
        return 1

    # Print results
    summary_df = pd.DataFrame(summaries).sort_values("saving_gbp", ascending=False)
    write_csv(summary_df, out_dir / "optimised_charging_summary.csv")

    headers = ["tariff", "total_battery_charge_kwh", "actual_charging_cost_gbp", "optimised_charging_cost_gbp", "saving_gbp", "saving_pct"]
    widths = {h: max(len(h), max(len(str(r.get(h, ""))) for r in summaries)) for h in headers}
    print("  ".join(h.ljust(widths[h]) for h in headers))
    print("  ".join("-" * widths[h] for h in headers))
    for row in summary_df.to_dict(orient="records"):
        print("  ".join(str(row.get(h, "")).ljust(widths[h]) for h in headers))

    best = summary_df.iloc[0]
    print(f"\nBiggest saving: {best['tariff']} — £{best['saving_gbp']:.2f} ({best['saving_pct']:.1f}%)")
    print(f"Outputs written to: {out_dir}")
    return 0


def run_best_tariff(args) -> int:
    """Run full battery simulation to find the absolute best tariff."""
    scenario = ScenarioConfig(
        battery_capacity_kwh=args.battery_capacity_kwh,
        extra_daily_kwh=getattr(args, "extra_daily_kwh", 0.0),
        extra_start=getattr(args, "extra_start", "17:00"),
        extra_end=getattr(args, "extra_end", "22:00"),
        ev_exclusion_enabled=not args.no_ev_exclusion,
    )

    power_csv = Path(args.power_csv)
    if not power_csv.exists():
        print(f"Error: {args.power_csv} not found.")
        return 1

    hh = load_power_csv(power_csv, scenario)
    hh = trim_date_range(hh, args.start_date, args.end_date)
    if hh.empty:
        raise RuntimeError("No power data left after applying date filter")

    num_days = hh["date"].nunique()
    total_solar = hh["solar_kwh"].sum()
    total_load = hh["load_kwh"].sum()
    total_import = hh["total_import_kwh"].sum()
    total_export = hh["export_kwh"].sum()

    print("=" * 70)
    print("BEST TARIFF ANALYSIS — Full Battery Optimisation Simulation")
    print("=" * 70)
    print(f"  Period: {num_days} days")
    print(f"  Solar generated: {total_solar:.1f} kWh ({total_solar/num_days:.1f} kWh/day)")
    print(f"  Home consumption: {total_load:.1f} kWh ({total_load/num_days:.1f} kWh/day)")
    print(f"  Actual grid import: {total_import:.1f} kWh")
    print(f"  Actual grid export: {total_export:.1f} kWh")
    print(f"  Battery: {args.battery_capacity_kwh} kWh capacity, "
          f"{args.battery_max_charge_kw}/{args.battery_max_discharge_kw} kW charge/discharge")
    print(f"  Efficiency: {args.battery_efficiency*100:.0f}% round-trip")
    print()

    # Determine which tariffs to simulate
    tariffs_to_run = getattr(args, "tariffs", None)
    all_tariffs = {"intelligent", "agile", "go", "flux", "cosy", "flexible", "tracker"}
    if tariffs_to_run:
        selected = {t.strip().lower() for t in tariffs_to_run.split(",")}
        invalid = selected - all_tariffs
        if invalid:
            print(f"Error: unknown tariff(s): {', '.join(sorted(invalid))}")
            return 1
    else:
        selected = all_tariffs

    # Build tariff configs
    tariff_configs = {}
    if "intelligent" in selected:
        tariff_configs["intelligent"] = IntelligentTariff(
            offpeak_import_p_per_kwh=args.intelligent_offpeak_import,
            peak_import_p_per_kwh=args.intelligent_peak_import,
            export_p_per_kwh=args.intelligent_export,
            standing_charge_p_per_day=args.intelligent_standing_charge,
            offpeak_start=args.intelligent_offpeak_start,
            offpeak_end=args.intelligent_offpeak_end,
        )
    if "go" in selected:
        tariff_configs["go"] = GoTariff(
            offpeak_import_p_per_kwh=getattr(args, "go_offpeak_import", 8.0),
            peak_import_p_per_kwh=getattr(args, "go_peak_import", 24.5),
            export_p_per_kwh=getattr(args, "go_export", 12.0),
            standing_charge_p_per_day=getattr(args, "go_standing_charge", 53.35),
            offpeak_start=getattr(args, "go_offpeak_start", "23:30"),
            offpeak_end=getattr(args, "go_offpeak_end", "05:30"),
        )
    if "flux" in selected:
        tariff_configs["flux"] = FluxTariff(
            offpeak_import_p_per_kwh=getattr(args, "flux_offpeak_import", 9.80),
            day_import_p_per_kwh=getattr(args, "flux_day_import", 22.36),
            peak_import_p_per_kwh=getattr(args, "flux_peak_import", 33.54),
            offpeak_export_p_per_kwh=getattr(args, "flux_offpeak_export", 4.05),
            day_export_p_per_kwh=getattr(args, "flux_day_export", 14.40),
            peak_export_p_per_kwh=getattr(args, "flux_peak_export", 28.60),
            standing_charge_p_per_day=getattr(args, "flux_standing_charge", 48.93),
            offpeak_start=getattr(args, "flux_offpeak_start", "02:00"),
            offpeak_end=getattr(args, "flux_offpeak_end", "05:00"),
            peak_start=getattr(args, "flux_peak_start", "16:00"),
            peak_end=getattr(args, "flux_peak_end", "19:00"),
        )
    if "cosy" in selected:
        tariff_configs["cosy"] = CosyTariff(
            cosy_import_p_per_kwh=getattr(args, "cosy_import", 12.0),
            day_import_p_per_kwh=getattr(args, "cosy_day_import", 24.50),
            peak_import_p_per_kwh=getattr(args, "cosy_peak_import", 36.75),
            export_p_per_kwh=getattr(args, "cosy_export", 12.0),
            standing_charge_p_per_day=getattr(args, "cosy_standing_charge", 53.35),
        )
    if "flexible" in selected:
        tariff_configs["flexible"] = FlexibleTariff(
            import_p_per_kwh=getattr(args, "flexible_import", 24.50),
            export_p_per_kwh=getattr(args, "flexible_export", 12.0),
            standing_charge_p_per_day=getattr(args, "flexible_standing_charge", 53.35),
        )

    # Fetch variable rate data
    agile_import_rates = None
    agile_export_rates = None
    tracker_rates = None

    if "agile" in selected:
        try:
            in_path, out_path = download_region_tariffs(
                args.region_code, Path(args.cache_dir), force=args.refresh_tariffs
            )
            agile_import_rates = read_agile_csv(in_path)
            agile_export_rates = read_agile_csv(out_path)
            tariff_configs["agile"] = AgileConfig(
                standing_charge_p_per_day=getattr(args, "agile_standing_charge", 66.26),
            )
        except (FileNotFoundError, requests.RequestException) as e:
            print(f"  Warning: could not fetch Agile rates: {e}")

    if "tracker" in selected:
        tracker_rates = download_tracker_rates(
            args.region_code, Path(args.cache_dir), force=args.refresh_tariffs
        )
        if tracker_rates is not None and not tracker_rates.empty:
            tariff_configs["tracker"] = TrackerTariff(
                standing_charge_p_per_day=getattr(args, "tracker_standing_charge", 53.35),
                export_p_per_kwh=getattr(args, "tracker_export", 12.0),
            )
        else:
            print("  Warning: could not fetch Tracker rates, skipping")

    # Run simulation for each tariff
    summaries = []
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for tariff_name, tariff_cfg in tariff_configs.items():
        s, daily = simulate_optimal_battery(
            hh, tariff_name, tariff_cfg, scenario,
            agile_import_rates=agile_import_rates,
            agile_export_rates=agile_export_rates,
            tracker_rates=tracker_rates,
            battery_max_charge_kw=args.battery_max_charge_kw,
            battery_max_discharge_kw=args.battery_max_discharge_kw,
            battery_efficiency=args.battery_efficiency,
        )
        if s is not None:
            summaries.append(s)
            write_csv(daily, out_dir / f"best_tariff_daily_{tariff_name}.csv")

    if not summaries:
        print("No tariff data available for simulation.")
        return 1

    # Sort by optimised net cost (the true best tariff)
    summary_df = pd.DataFrame(summaries).sort_values("optimised_net_gbp")
    write_csv(summary_df, out_dir / "best_tariff_summary.csv")

    # Print results
    print("With optimal battery scheduling, your costs would be:")
    print()
    headers = ["tariff", "optimised_import_gbp", "optimised_export_gbp",
               "actual_standing_gbp", "optimised_net_gbp"]
    display_headers = ["tariff", "import_£", "export_£", "standing_£", "net_cost_£"]
    rows = summary_df.to_dict(orient="records")
    widths = {dh: max(len(dh), max(len(str(r.get(h, ""))) for r in rows))
              for h, dh in zip(headers, display_headers, strict=True)}
    print("  ".join(dh.ljust(widths[dh]) for dh in display_headers))
    print("  ".join("-" * widths[dh] for dh in display_headers))
    for row in rows:
        vals = [str(row.get(h, "")) for h in headers]
        print("  ".join(v.ljust(widths[dh]) for v, dh in zip(vals, display_headers, strict=True)))

    best_row = rows[0]
    print(f"\n{'='*70}")
    print(f"  BEST TARIFF: {best_row['tariff'].upper()}")
    print(f"  Optimised net cost: £{best_row['optimised_net_gbp']:.2f} "
          f"over {best_row['days']} days "
          f"(£{best_row['optimised_net_gbp']/best_row['days']*365:.0f}/year)")
    if best_row["saving_gbp"] > 0:
        print(f"  Saving vs actual: £{best_row['saving_gbp']:.2f} "
              f"({best_row['saving_pct']:.1f}%)")
    print(f"{'='*70}")

    # Also show comparison between tariffs
    if len(rows) > 1:
        worst_row = rows[-1]
        spread = worst_row["optimised_net_gbp"] - best_row["optimised_net_gbp"]
        print(f"\n  Spread between best and worst: £{spread:.2f} "
              f"(£{spread/best_row['days']*365:.0f}/year)")

    print(f"\nOutputs written to: {out_dir}")
    return 0


def _format_tz_date(dt) -> str:
    """Format a datetime with proper ISO timezone (e.g. +01:00 not +0100)."""
    s = dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    return re.sub(r'([+-])(\d{2})(\d{2})$', r'\1\2:\3', s)


def run_download_data(args) -> int:
    """Download Powerwall 5-minute power data day-by-day, merge, and run comparison."""
    import time as _time
    import traceback
    from datetime import datetime, timedelta

    try:
        import teslapy
    except ImportError:
        import sys as _sys
        print("Error: teslapy library not installed. Run: pip install teslapy", file=_sys.stderr)
        return 1

    print(f"Logging in to Tesla as {args.email}...")
    tesla = teslapy.Tesla(args.email, retry=2, timeout=10)
    # Tesla deprecated the https://auth.tesla.com/void/callback redirect URI;
    # the Tesla app's tesla://auth/callback is the only redirect still
    # registered for the ownerapi client_id.
    tesla.redirect_uri = "tesla://auth/callback"

    if not tesla.authorized:
        print("STEP 1: Log in to Tesla. Open this page in your browser:\n")
        print(tesla.authorization_url())
        print()
        print("After successful login, you will see a 'Verified Successfully' page.")
        print("Most browsers won't navigate to the tesla:// URL, so you need to")
        print("copy it from the browser's developer console:")
        print("  1. Open developer tools and switch to the Console tab")
        print("     (Chrome: View > Developer > Developer Tools)")
        print('  2. Find the message "Failed to launch \'tesla://auth/callback?...\'" or similar')
        print("  3. Right-click the tesla://auth/callback?... URL and choose 'Copy link address'")
        print()
        print("The URL should look like: tesla://auth/callback?code=NA_abcd12345...&issuer=...")
        print()
        auth_response = input("Paste the URL here: ")
        # oauthlib refuses to parse non-https authorization responses. Only the
        # code/state query params are read from this URL, so rewriting the
        # scheme is safe.
        auth_response = auth_response.replace(
            "tesla://auth/callback", "https://auth.tesla.com/void/callback", 1
        )
        tesla.fetch_token(authorization_response=auth_response)
        print("\nSuccess!")

    for product in tesla.api("PRODUCT_LIST")["response"]:
        resource_type = product.get("resource_type")
        if resource_type not in ("battery", "solar"):
            continue

        site_id = product["energy_site_id"]
        print(f"\nFound {resource_type} site {site_id}")

        # Get site config for timezone and installation date
        site_config = tesla.api("SITE_CONFIG", path_vars={"site_id": site_id})["response"]
        timezone = site_config.get("installation_time_zone", "Europe/London")
        tz = ZoneInfo(timezone)

        installation_date_str = site_config.get("installation_date", "")
        if installation_date_str:
            installation_date = pd.Timestamp(installation_date_str)
            if installation_date.tzinfo is None:
                installation_date = installation_date.tz_localize(tz)
            else:
                installation_date = installation_date.tz_convert(tz)
        else:
            installation_date = pd.Timestamp.now(tz) - pd.Timedelta(days=365)

        # Limit to 1 year back or installation date, whichever is more recent
        one_year_ago = pd.Timestamp.now(tz) - pd.Timedelta(days=365)
        earliest = max(installation_date, one_year_ago)

        power_dir = Path("download/power")
        power_dir.mkdir(parents=True, exist_ok=True)

        # Download day-by-day, skipping days already on disk
        now = datetime.now(tz)
        current_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        partial_day = True  # today is always partial

        # Count how many days need downloading
        days_to_check = []
        check_day = current_day
        while check_day >= earliest:
            date_str = check_day.strftime("%Y-%m-%d")
            csv_path = power_dir / f"{date_str}.csv"
            if not csv_path.exists():
                days_to_check.append(check_day)
            check_day -= timedelta(days=1)
            check_day = check_day.replace(tzinfo=None).replace(tzinfo=tz)

        # Always re-download today
        days_needing_download = len(days_to_check)
        if days_needing_download == 0:
            days_needing_download = 1  # at least today's partial

        total_days = (current_day - earliest).days + 1
        existing_days = total_days - len(days_to_check)

        if existing_days == 0:
            est_minutes = (days_needing_download * 1.5) / 60
            print(f"  No existing data found. Downloading {days_needing_download} days.")
            print(f"  Estimated time: ~{est_minutes:.0f} minutes")
        elif days_needing_download <= 1:
            print(f"  Data up to date ({existing_days} days cached). Refreshing today...")
        else:
            est_minutes = (days_needing_download * 1.5) / 60
            print(f"  {existing_days} days cached, {days_needing_download} days to download.")
            print(f"  Estimated time: ~{est_minutes:.0f} minutes")

        print(f"  Date range: {earliest.strftime('%Y-%m-%d')} to {current_day.strftime('%Y-%m-%d')}")

        days_downloaded = 0
        days_skipped = 0

        while current_day >= earliest:
            date_str = current_day.strftime("%Y-%m-%d")
            csv_path = power_dir / f"{date_str}.csv"

            # Skip if already downloaded (unless it's today's partial file)
            if not partial_day and csv_path.exists():
                days_skipped += 1
                current_day -= timedelta(days=1)
                current_day = current_day.replace(tzinfo=None).replace(tzinfo=tz)
                continue

            # Remove stale partial file for today
            partial_path = power_dir / f"{date_str}.partial.csv"
            if partial_day:
                partial_path.unlink(missing_ok=True)
                csv_path.unlink(missing_ok=True)

            print(f"    {date_str}{'  (partial)' if partial_day else ''}")

            try:
                day_start = current_day.replace(hour=0, minute=0, second=0, microsecond=0)
                day_end = current_day.replace(hour=23, minute=59, second=59, microsecond=0)

                response = tesla.api(
                    "CALENDAR_HISTORY_DATA",
                    path_vars={"site_id": site_id},
                    kind="power",
                    period="day",
                    start_date=_format_tz_date(day_start),
                    end_date=_format_tz_date(day_end),
                    time_zone=timezone,
                )

                if response and "time_series" in response["response"]:
                    ts = response["response"]["time_series"]
                    for row in ts:
                        row["load_power"] = (
                            row["solar_power"] + row["battery_power"]
                            + row["grid_power"] + row.get("generator_power", 0)
                        )
                    df_day = pd.DataFrame(ts)
                    save_path = partial_path if partial_day else csv_path
                    df_day.to_csv(save_path, index=False)
                    days_downloaded += 1
            except Exception:
                traceback.print_exc()

            _time.sleep(1)
            partial_day = False
            current_day -= timedelta(days=1)
            current_day = current_day.replace(tzinfo=None).replace(tzinfo=tz)

        print(f"\n  Done: {days_downloaded} days downloaded, {days_skipped} skipped (already on disk)")

        # Merge all per-day CSVs into one combined file
        print("  Merging into download/power.csv...")
        csv_files = sorted(power_dir.glob("*.csv"))
        if not csv_files:
            print("  No data files found.")
            continue

        frames = []
        for f in csv_files:
            try:
                frames.append(pd.read_csv(f))
            except Exception:
                print(f"    Warning: could not read {f.name}, skipping")

        if not frames:
            print("  No valid data files found.")
            continue

        combined = pd.concat(frames, ignore_index=True)
        combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)
        combined = combined.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
        combined_path = Path("download/power.csv")
        combined.to_csv(combined_path, index=False)
        print(f"  Saved {len(combined)} rows to {combined_path}")
        print(f"  Date range: {combined['timestamp'].min()} to {combined['timestamp'].max()}")

    print("\nDownload complete!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
