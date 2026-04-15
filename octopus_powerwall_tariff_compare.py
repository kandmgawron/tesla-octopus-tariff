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


def download_region_tariffs(region_code: str, cache_dir: Path, index_url: str = DEFAULT_TARIFF_INDEX, force: bool = False) -> Tuple[Path, Path]:
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

    if scenario.ev_exclusion_enabled:
        df["is_car_charging"] = df.apply(
            lambda row: row["grid_power"] >= scenario.ev_min_power_w and time_in_window(row["timestamp_local"], scenario.ev_start, scenario.ev_end),
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


def calculate_agile(hh: pd.DataFrame, agile_import: pd.DataFrame, agile_export: pd.DataFrame, config: AgileConfig, scenario: ScenarioConfig):
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
    scenario = ScenarioConfig(
        battery_capacity_kwh=args.battery_capacity_kwh,
        extra_daily_kwh=args.extra_daily_kwh,
        extra_start=args.extra_start,
        extra_end=args.extra_end,
        ev_exclusion_enabled=not args.no_ev_exclusion,
    )
    intelligent = IntelligentTariff(
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

    hh = load_power_csv(Path(args.power_csv), scenario)
    hh = trim_date_range(hh, args.start_date, args.end_date)
    if hh.empty:
        raise RuntimeError("No power data left after applying date filter")

    in_path, out_path = download_region_tariffs(args.region_code, Path(args.cache_dir), force=args.refresh_tariffs)
    agile_in = read_agile_csv(in_path)
    agile_out = read_agile_csv(out_path)

    intelligent_summary, intelligent_daily = calculate_intelligent(hh, intelligent, scenario)
    agile_summary, agile_daily = calculate_agile(hh, agile_in, agile_out, agile_cfg, scenario)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame([intelligent_summary, agile_summary]).sort_values("net_cost_gbp")
    write_csv(summary_df, out_dir / "summary.csv")
    write_csv(intelligent_daily, out_dir / "daily_breakdown_intelligent.csv")
    write_csv(agile_daily, out_dir / "daily_breakdown_agile.csv")

    rows = summary_df.to_dict(orient="records")
    print_summary_table(rows)
    print(f"\nBest tariff by model: {rows[0]['tariff']}")
    print(f"Outputs written to: {out_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare Tesla Powerwall usage against Octopus tariffs with sane defaults.",
        epilog="Use --download-data to automatically download Powerwall data using teslapy. "
               "This will prompt for Tesla login and save power data to download/ directory.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    default = sub.add_parser("default", help="Simplest first run")
    default.add_argument("--power-csv", required=True, help="Tesla power_all.csv style file")
    default.add_argument("--out-dir", default="output")
    default.add_argument("--cache-dir", default=".cache/tariffs")
    default.add_argument("--region-code", default="M")
    default.add_argument("--extra-daily-kwh", type=float, default=0.0)
    default.add_argument("--battery-capacity-kwh", type=float, default=13.0)
    default.add_argument("--start-date")
    default.add_argument("--end-date")
    default.add_argument("--refresh-tariffs", action="store_true")
    default.add_argument("--no-ev-exclusion", action="store_true")
    default.set_defaults(
        intelligent_offpeak_import=7.0,
        intelligent_peak_import=26.0,
        intelligent_export=15.0,
        intelligent_standing_charge=57.01,
        intelligent_offpeak_start="23:30",
        intelligent_offpeak_end="05:30",
        agile_standing_charge=66.26,
        agile_flex_hours=6.0,
        agile_flex_max_kw=3.3,
        extra_start="17:00",
        extra_end="22:00",
    )

    compare = sub.add_parser("compare", help="Advanced mode with more overrides")
    compare.add_argument("--power-csv", required=True)
    compare.add_argument("--out-dir", default="output")
    compare.add_argument("--cache-dir", default=".cache/tariffs")
    compare.add_argument("--region-code", default="M")
    compare.add_argument("--start-date")
    compare.add_argument("--end-date")
    compare.add_argument("--refresh-tariffs", action="store_true")
    compare.add_argument("--battery-capacity-kwh", type=float, default=13.0)
    compare.add_argument("--extra-daily-kwh", type=float, default=0.0)
    compare.add_argument("--extra-start", default="17:00")
    compare.add_argument("--extra-end", default="22:00")
    compare.add_argument("--intelligent-offpeak-import", type=float, default=7.0)
    compare.add_argument("--intelligent-peak-import", type=float, default=26.0)
    compare.add_argument("--intelligent-export", type=float, default=15.0)
    compare.add_argument("--intelligent-standing-charge", type=float, default=57.01)
    compare.add_argument("--intelligent-offpeak-start", default="23:30")
    compare.add_argument("--intelligent-offpeak-end", default="05:30")
    compare.add_argument("--agile-standing-charge", type=float, default=66.26)
    compare.add_argument("--agile-flex-hours", type=float, default=6.0)
    compare.add_argument("--agile-flex-max-kw", type=float, default=3.3)
    compare.add_argument("--no-ev-exclusion", action="store_true")

    download = sub.add_parser("download-data", help="Download Powerwall data using teslapy")
    download.add_argument("--email", required=True, help="Tesla account email address")

    regions = sub.add_parser("list-regions", help="Show supported region codes")
    regions.add_argument("--json", action="store_true")

    refresh = sub.add_parser("refresh-tariffs", help="Download the latest region tariff CSVs into cache")
    refresh.add_argument("--region-code", default="M")
    refresh.add_argument("--cache-dir", default=".cache/tariffs")

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "download-data":
        return run_download_data(args)
    if args.command in {"default", "compare"}:
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
        print("Error: teslapy library not installed. Run: pip install teslapy", file=sys.stderr)
        return 1

    print(f"Logging in to Tesla as {args.email}...")
    tesla = teslapy.Tesla(args.email, retry=2, timeout=10)

    if not tesla.authorized:
        print("STEP 1: Log in to Tesla. Open this page in your browser:\n")
        print(tesla.authorization_url())
        print()
        print("After successful login, you will get a Page Not Found error. That's expected.")
        print("Just copy the url of that page and paste it here:")
        tesla.fetch_token(authorization_response=input("URL after authentication: "))
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

        power_dir = Path(f"download/{site_id}/power")
        power_dir.mkdir(parents=True, exist_ok=True)

        # Download day-by-day, skipping days already on disk
        now = datetime.now(tz)
        current_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        partial_day = True  # today is always partial
        days_downloaded = 0
        days_skipped = 0

        print(f"  Downloading 5-minute power data from {earliest.strftime('%Y-%m-%d')} to {current_day.strftime('%Y-%m-%d')}...")

        while current_day >= earliest:
            date_str = current_day.strftime("%Y-%m-%d")
            csv_path = power_dir / f"{date_str}.csv"

            # Skip if already downloaded (unless it's today's partial file)
            if not partial_day and csv_path.exists():
                days_skipped += 1
                current_day -= timedelta(days=1)
                current_day = current_day.replace(tzinfo=None)
                current_day = tz.localize(current_day) if hasattr(tz, 'localize') else current_day.replace(tzinfo=tz)
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
            # Re-localize after subtracting to handle DST transitions
            current_day = current_day.replace(tzinfo=None)
            current_day = current_day.replace(tzinfo=tz)

        print(f"\n  Done: {days_downloaded} days downloaded, {days_skipped} skipped (already on disk)")

        # Merge all per-day CSVs into one combined file
        print("  Merging per-day files into combined power.csv...")
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
        combined_path = Path(f"download/{site_id}/power.csv")
        combined.to_csv(combined_path, index=False)
        print(f"  Saved {len(combined)} rows to {combined_path}")
        print(f"  Date range: {combined['timestamp'].min()} to {combined['timestamp'].max()}")

        # Clean up per-day files now that the merged CSV exists
        print("  Cleaning up per-day files...")
        import shutil
        shutil.rmtree(power_dir)
        print(f"  Removed {power_dir}")

    print("\nDownload complete!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
