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
    "agile_standing_charge",
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


@dataclass
class ScenarioConfig:
    battery_capacity_kwh: float = 13.0
    extra_daily_kwh: float = 0.0
    extra_start: str = "17:00"
    extra_end: str = "22:00"
    timezone: str = "Europe/London"


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
    df["slot_start"] = df["timestamp"].dt.floor("30min").dt.tz_convert(tz)

    df["import_kwh_5m"] = np.where(df["grid_power"] > 0, df["grid_power"] / 12.0 / 1000.0, 0.0)
    df["export_kwh_5m"] = np.where(df["grid_power"] < 0, -df["grid_power"] / 12.0 / 1000.0, 0.0)

    if "solar_power" in df.columns:
        df["solar_power"] = pd.to_numeric(df["solar_power"], errors="coerce").fillna(0.0)
        df["solar_kwh_5m"] = np.maximum(df["solar_power"], 0.0) / 12.0 / 1000.0
    else:
        df["solar_kwh_5m"] = 0.0

    if "load_power" in df.columns:
        df["load_power"] = pd.to_numeric(df["load_power"], errors="coerce").fillna(0.0)
        df["load_kwh_5m"] = np.maximum(df["load_power"], 0.0) / 12.0 / 1000.0
    else:
        df["load_kwh_5m"] = df["import_kwh_5m"]

    hh = (
        df.groupby("slot_start", as_index=False)
        .agg(
            total_import_kwh=("import_kwh_5m", "sum"),
            export_kwh=("export_kwh_5m", "sum"),
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


# ── Rate lookup helpers ───────────────────────────────────────────────────────


def _minutes_in_window(minutes: np.ndarray, start_hhmm: str, end_hhmm: str) -> np.ndarray:
    """Vectorised version of `time_in_window` for a numpy array of minute-of-day values."""
    sh, sm = parse_hhmm(start_hhmm)
    eh, em = parse_hhmm(end_hhmm)
    smins = sh * 60 + sm
    emins = eh * 60 + em
    if smins < emins:
        return (minutes >= smins) & (minutes < emins)
    return (minutes >= smins) | (minutes < emins)


def compute_tariff_rates(
    slots: pd.Series,
    tariff_name: str,
    tariff_config,
    agile_import_rates: Optional[pd.DataFrame] = None,
    agile_export_rates: Optional[pd.DataFrame] = None,
    tracker_rates: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (import_rates_p, export_rates_p) numpy arrays for all slots.

    Vectorised replacement for the previous per-row rate-lookup helpers. NaN values
    indicate that no rate was available for that slot.
    """
    n = len(slots)
    minutes = (slots.dt.hour * 60 + slots.dt.minute).to_numpy()
    default_export = getattr(tariff_config, "export_p_per_kwh", 12.0)

    if tariff_name in ("intelligent", "go"):
        offpeak = _minutes_in_window(minutes, tariff_config.offpeak_start, tariff_config.offpeak_end)
        import_rates = np.where(
            offpeak,
            tariff_config.offpeak_import_p_per_kwh,
            tariff_config.peak_import_p_per_kwh,
        ).astype(float)
        export_rates = np.full(n, default_export, dtype=float)
        return import_rates, export_rates

    if tariff_name == "flux":
        offpeak = _minutes_in_window(minutes, tariff_config.offpeak_start, tariff_config.offpeak_end)
        peak = _minutes_in_window(minutes, tariff_config.peak_start, tariff_config.peak_end)
        # Offpeak takes priority over peak when bands overlap.
        import_rates = np.where(
            offpeak,
            tariff_config.offpeak_import_p_per_kwh,
            np.where(peak, tariff_config.peak_import_p_per_kwh, tariff_config.day_import_p_per_kwh),
        ).astype(float)
        export_rates = np.where(
            offpeak,
            tariff_config.offpeak_export_p_per_kwh,
            np.where(peak, tariff_config.peak_export_p_per_kwh, tariff_config.day_export_p_per_kwh),
        ).astype(float)
        return import_rates, export_rates

    if tariff_name == "cosy":
        peak = _minutes_in_window(minutes, tariff_config.peak_start, tariff_config.peak_end)
        cosy = (
            _minutes_in_window(minutes, tariff_config.cosy_window_1_start, tariff_config.cosy_window_1_end)
            | _minutes_in_window(minutes, tariff_config.cosy_window_2_start, tariff_config.cosy_window_2_end)
            | _minutes_in_window(minutes, tariff_config.cosy_window_3_start, tariff_config.cosy_window_3_end)
        )
        # Peak takes priority over the cosy windows; remaining slots get the day rate.
        import_rates = np.where(
            peak,
            tariff_config.peak_import_p_per_kwh,
            np.where(cosy, tariff_config.cosy_import_p_per_kwh, tariff_config.day_import_p_per_kwh),
        ).astype(float)
        export_rates = np.full(n, default_export, dtype=float)
        return import_rates, export_rates

    if tariff_name == "flexible":
        import_rates = np.full(n, tariff_config.import_p_per_kwh, dtype=float)
        export_rates = np.full(n, default_export, dtype=float)
        return import_rates, export_rates

    if tariff_name == "agile":
        import_rates = np.full(n, np.nan, dtype=float)
        export_rates = np.full(n, np.nan, dtype=float)
        slot_df = slots.reset_index(drop=True).to_frame("slot_start")
        if agile_import_rates is not None:
            merged = slot_df.merge(
                agile_import_rates[["slot_start", "price_p_per_kwh"]],
                on="slot_start", how="left",
            )
            import_rates = merged["price_p_per_kwh"].to_numpy(dtype=float)
        if agile_export_rates is not None:
            merged = slot_df.merge(
                agile_export_rates[["slot_start", "price_p_per_kwh"]],
                on="slot_start", how="left",
            )
            export_rates = merged["price_p_per_kwh"].to_numpy(dtype=float)
        return import_rates, export_rates

    if tariff_name == "tracker":
        import_rates = np.full(n, np.nan, dtype=float)
        export_rates = np.full(n, default_export, dtype=float)
        if tracker_rates is not None:
            dates = pd.Series(list(slots.dt.date), name="date")
            merged = dates.to_frame().merge(
                tracker_rates[["date", "price_p_per_kwh"]],
                on="date", how="left",
            )
            import_rates = merged["price_p_per_kwh"].to_numpy(dtype=float)
        return import_rates, export_rates

    return np.full(n, np.nan, dtype=float), np.full(n, np.nan, dtype=float)


# ── Tracker tariff fetcher ────────────────────────────────────────────────────


def download_tracker_rates(region_code: str, cache_dir: Path, force: bool = False) -> Optional[pd.DataFrame]:
    """Download Octopus Tracker daily rates from the Octopus API.

    Returns a DataFrame with columns: date, price_p_per_kwh
    Returns None if rates cannot be fetched.
    """
    region_code = region_code.upper()
    cache_path = cache_dir / f"tracker_{region_code}.csv"

    if not force and cache_path.exists():
        df = pd.read_csv(cache_path)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    def _fetch_tracker_pages(product_code: str) -> list:
        """Fetch all paginated tracker rate results for a product code."""
        tariff_code = f"E-1R-{product_code}-{region_code}"
        page_url = (
            f"https://api.octopus.energy/v1/products/{product_code}"
            f"/electricity-tariffs/{tariff_code}/standard-unit-rates/"
        )
        results: list = []
        while page_url:
            resp = requests.get(page_url, timeout=30, params={"page_size": 1500})
            resp.raise_for_status()
            data = resp.json()
            results.extend(data.get("results", []))
            page_url = data.get("next")
        return results

    try:
        all_results = _fetch_tracker_pages("SILVER-24-04-03")
    except requests.RequestException as e:
        print(f"  Warning: could not fetch Tracker rates: {e}")
        try:
            all_results = _fetch_tracker_pages("SILVER-FLEX-22-11-25")
        except requests.RequestException as e2:
            print(f"  Warning: fallback Tracker fetch also failed: {e2}")
            return None

    if not all_results:
        print("  Warning: no Tracker rate data returned from API")
        return None

    df = pd.DataFrame(all_results)
    df["valid_from"] = pd.to_datetime(df["valid_from"], utc=True)
    df["date"] = df["valid_from"].dt.tz_convert(LOCAL_TZ).dt.date
    df = df.sort_values("valid_from").drop_duplicates(subset=["date"], keep="last")
    df = df.rename(columns={"value_inc_vat": "price_p_per_kwh"})
    df = df[["date", "price_p_per_kwh"]].sort_values("date")

    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


# ── Output helpers ────────────────────────────────────────────────────────────


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def print_summary_table(rows, headers=None):
    if not rows:
        return
    if headers is None:
        headers = list(rows[0].keys())
    widths = {h: max(len(h), max(len(str(r.get(h, ""))) for r in rows)) for h in headers}
    print("  ".join(h.ljust(widths[h]) for h in headers))
    print("  ".join("-" * widths[h] for h in headers))
    for row in rows:
        print("  ".join(str(row.get(h, "")).ljust(widths[h]) for h in headers))


def _format_tz_date(dt) -> str:
    """Format a datetime with proper ISO timezone (e.g. +01:00 not +0100)."""
    s = dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    return re.sub(r'([+-])(\d{2})(\d{2})$', r'\1\2:\3', s)


# ── Main Simulation ───────────────────────────────────────────────────────────


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
    """Simulate optimal battery usage with corrected solar export logic.

    For each slot the solar decision depends on rate comparison:
    - If export_rate > import_rate: export ALL solar (buy from grid for load — net cheaper)
    - If export_rate <= import_rate: solar serves load first, surplus to battery/export

    Battery strategy:
    - Charge from grid when import rate is below the daily charge threshold
    - Discharge to serve load when import rate is above threshold
    - Discharge to export when export rate is above threshold (keep 10% reserve)
    - Charge threshold = rate at the slot where battery would be full from cheapest slots
    """
    df = hh.copy()
    df["date"] = df["slot_start"].dt.date

    # Add extra load profile if configured
    df["extra_kwh"] = build_extra_profile(
        df["slot_start"], scenario.extra_daily_kwh, scenario.extra_start, scenario.extra_end
    )

    # Get import and export rates for each slot (vectorised)
    import_rates_arr, export_rates_arr = compute_tariff_rates(
        df["slot_start"], tariff_name, tariff_config,
        agile_import_rates=agile_import_rates,
        agile_export_rates=agile_export_rates,
        tracker_rates=tracker_rates,
    )
    df["import_rate_p"] = import_rates_arr
    df["export_rate_p"] = export_rates_arr

    # Drop slots with missing rate data
    df = df.dropna(subset=["import_rate_p", "export_rate_p"]).copy()
    if df.empty:
        return None, None

    slot_max_charge_kwh = battery_max_charge_kw * 0.5
    slot_max_discharge_kwh = battery_max_discharge_kw * 0.5
    battery_cap = scenario.battery_capacity_kwh
    min_soc = battery_cap * 0.10  # 10% reserve

    daily_rows = []
    total_opt_import_cost_p = 0.0
    total_opt_export_revenue_p = 0.0
    total_days = 0

    for day, group in df.groupby("date", sort=True):
        group = group.copy().sort_values("slot_start").reset_index(drop=True)
        n_slots = len(group)
        if n_slots == 0:
            continue
        total_days += 1

        solar = group["solar_kwh"].values
        load = group["load_kwh"].values + group["extra_kwh"].values
        import_rates = group["import_rate_p"].values
        export_rates = group["export_rate_p"].values

        # Determine charge threshold: the rate at the boundary where battery fills
        # from the cheapest slots. Slots below this are "cheap" (charge), above are "expensive" (discharge).
        sorted_rates = np.sort(import_rates)
        slots_to_fill = int(np.ceil(battery_cap / slot_max_charge_kwh))
        if slots_to_fill < n_slots:
            charge_threshold = sorted_rates[min(slots_to_fill, n_slots - 1)]
        else:
            charge_threshold = sorted_rates[-1]

        # Simulate slot by slot
        soc = battery_cap * 0.5  # Start day at 50%
        opt_import_cost_p = 0.0
        opt_export_revenue_p = 0.0
        opt_grid_import_kwh = 0.0
        opt_grid_export_kwh = 0.0

        for i in range(n_slots):
            slot_solar = solar[i]
            slot_load = load[i]
            slot_import_rate = import_rates[i]
            slot_export_rate = export_rates[i]

            # ── Solar decision ──
            # If export rate > import rate: export ALL solar, serve load from battery/grid
            # Otherwise: solar serves load first, surplus to battery or export
            solar_to_load = 0.0
            solar_to_battery = 0.0
            solar_to_export = 0.0
            remaining_load = slot_load

            if slot_export_rate > slot_import_rate:
                # More profitable to export all solar and buy from grid
                solar_to_export = slot_solar
            else:
                # Solar serves load first (avoids expensive import)
                solar_to_load = min(slot_solar, slot_load)
                remaining_solar = slot_solar - solar_to_load
                remaining_load = slot_load - solar_to_load

                # Surplus solar charges battery, then exports
                if remaining_solar > 0:
                    can_charge = min(remaining_solar, slot_max_charge_kwh, (battery_cap - soc) / battery_efficiency)
                    can_charge = max(can_charge, 0.0)
                    solar_to_battery = can_charge
                    soc += solar_to_battery * battery_efficiency
                    solar_to_export = remaining_solar - solar_to_battery

            # ── Serve remaining load: battery discharge or grid import ──
            battery_to_load = 0.0
            grid_to_load = 0.0
            if remaining_load > 0:
                # Discharge battery if import rate is expensive (above threshold)
                if slot_import_rate >= charge_threshold and soc > min_soc:
                    can_discharge = min(remaining_load, slot_max_discharge_kwh, soc - min_soc)
                    battery_to_load = can_discharge
                    soc -= battery_to_load
                    remaining_load -= battery_to_load
                grid_to_load = remaining_load

            # ── Charge battery from grid during cheap slots ──
            grid_to_battery = 0.0
            if slot_import_rate < charge_threshold and soc < battery_cap:
                available_charge = min(
                    slot_max_charge_kwh - solar_to_battery,
                    (battery_cap - soc) / battery_efficiency,
                )
                available_charge = max(available_charge, 0.0)
                if available_charge > 0.01:
                    grid_to_battery = available_charge
                    soc += grid_to_battery * battery_efficiency

            # ── Discharge battery to export when export rate is high ──
            battery_to_export = 0.0
            if slot_export_rate > charge_threshold and soc > min_soc:
                available_discharge = min(
                    slot_max_discharge_kwh - battery_to_load,
                    soc - min_soc,
                )
                available_discharge = max(available_discharge, 0.0)
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

        total_opt_import_cost_p += opt_import_cost_p
        total_opt_export_revenue_p += opt_export_revenue_p

        opt_net_p = opt_import_cost_p - opt_export_revenue_p

        daily_rows.append({
            "date": day,
            "solar_kwh": round(float(solar.sum()), 2),
            "load_kwh": round(float(load.sum()), 2),
            "optimised_import_cost_p": round(opt_import_cost_p, 2),
            "optimised_export_revenue_p": round(opt_export_revenue_p, 2),
            "optimised_net_cost_p": round(opt_net_p, 2),
            "opt_grid_import_kwh": round(opt_grid_import_kwh, 2),
            "opt_grid_export_kwh": round(opt_grid_export_kwh, 2),
        })

    if total_days == 0:
        return None, None

    # Get standing charge
    sc_p_per_day = 53.35
    if hasattr(tariff_config, "standing_charge_p_per_day"):
        sc_p_per_day = tariff_config.standing_charge_p_per_day
    total_sc_p = sc_p_per_day * total_days

    opt_net_total_p = total_opt_import_cost_p - total_opt_export_revenue_p + total_sc_p

    summary = {
        "tariff": tariff_name,
        "days": total_days,
        "optimised_import_gbp": round(pence_to_pounds(total_opt_import_cost_p), 2),
        "optimised_export_gbp": round(pence_to_pounds(total_opt_export_revenue_p), 2),
        "standing_charge_gbp": round(pence_to_pounds(total_sc_p), 2),
        "optimised_net_gbp": round(pence_to_pounds(opt_net_total_p), 2),
    }
    return summary, pd.DataFrame(daily_rows)


# ── Command implementations ───────────────────────────────────────────────────


def run_analyse(args) -> int:
    """Run full battery optimisation simulation across all tariffs."""
    scenario = ScenarioConfig(
        battery_capacity_kwh=args.battery_capacity_kwh,
        extra_daily_kwh=getattr(args, "extra_daily_kwh", 0.0),
        extra_start=getattr(args, "extra_start", "17:00"),
        extra_end=getattr(args, "extra_end", "22:00"),
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

    print("=" * 70)
    print("TARIFF ANALYSIS — Full Battery Optimisation Simulation")
    print("=" * 70)
    print(f"  Period: {num_days} days")
    print(f"  Solar generated: {total_solar:.1f} kWh ({total_solar / max(num_days, 1):.1f} kWh/day)")
    print(f"  Home consumption: {total_load:.1f} kWh ({total_load / max(num_days, 1):.1f} kWh/day)")
    print(f"  Battery: {args.battery_capacity_kwh} kWh capacity, "
          f"{args.battery_max_charge_kw}/{args.battery_max_discharge_kw} kW charge/discharge")
    print(f"  Efficiency: {args.battery_efficiency * 100:.0f}% round-trip")
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
            write_csv(daily, out_dir / f"daily_{tariff_name}.csv")

    if not summaries:
        print("No tariff data available for simulation.")
        return 1

    # Sort by optimised net cost
    summary_df = pd.DataFrame(summaries).sort_values("optimised_net_gbp")
    write_csv(summary_df, out_dir / "summary.csv")

    # Print results
    print("With optimal battery scheduling, your costs would be:")
    print()
    headers = ["tariff", "optimised_import_gbp", "optimised_export_gbp", "standing_charge_gbp", "optimised_net_gbp"]
    rows = summary_df.to_dict(orient="records")
    print_summary_table(rows, headers=headers)

    best_row = rows[0]
    print(f"\n{'=' * 70}")
    print(f"  BEST TARIFF: {best_row['tariff'].upper()}")
    days = best_row["days"]
    net = best_row["optimised_net_gbp"]
    annual = net / max(days, 1) * 365
    print(f"  Optimised net cost: \u00a3{net:.2f} over {days} days (\u00a3{annual:.0f}/year)")
    print(f"{'=' * 70}")

    if len(rows) > 1:
        worst_row = rows[-1]
        spread = worst_row["optimised_net_gbp"] - best_row["optimised_net_gbp"]
        annual_spread = spread / max(days, 1) * 365
        print(f"\n  Spread between best and worst: \u00a3{spread:.2f} (\u00a3{annual_spread:.0f}/year)")

    print(f"\nOutputs written to: {out_dir}")
    return 0


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
        auth_response = input("Paste the URL here: ")  # noqa: S322
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

        one_year_ago = pd.Timestamp.now(tz) - pd.Timedelta(days=365)
        earliest = max(installation_date, one_year_ago)

        power_dir = Path("download/power")
        power_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now(tz)
        current_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        partial_day = True

        days_to_check = []
        check_day = current_day
        while check_day >= earliest:
            date_str = check_day.strftime("%Y-%m-%d")
            csv_path = power_dir / f"{date_str}.csv"
            if not csv_path.exists():
                days_to_check.append(check_day)
            check_day -= timedelta(days=1)
            check_day = check_day.replace(tzinfo=None).replace(tzinfo=tz)

        days_needing_download = len(days_to_check)
        if days_needing_download == 0:
            days_needing_download = 1

        total_days_range = (current_day - earliest).days + 1
        existing_days = total_days_range - len(days_to_check)

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

            if not partial_day and csv_path.exists():
                days_skipped += 1
                current_day -= timedelta(days=1)
                current_day = current_day.replace(tzinfo=None).replace(tzinfo=tz)
                continue

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


def run_model(args) -> int:
    """Run scenario modelling with adjusted energy usage."""
    labels = args.labels
    adjust_kwhs = args.adjust_kwhs
    windows = args.windows

    if not (len(labels) == len(adjust_kwhs) == len(windows)):
        print("Error: --label, --adjust-kwh, and --window must be provided the same number of times")
        return 1

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

    # Run baseline
    print("=" * 60)
    print("BASELINE: Current usage")
    print("=" * 60)
    args.extra_daily_kwh = 0.0
    rc = run_analyse(args)
    if rc != 0:
        return rc

    out_dir = Path(args.out_dir)
    baseline_df = pd.read_csv(out_dir / "summary.csv")
    baseline_rows = baseline_df.to_dict(orient="records")

    # Run each scenario
    total_extra_kwh = 0.0
    scenario_descriptions = []

    for i, (label, kwh, window) in enumerate(zip(labels, adjust_kwhs, windows, strict=True)):
        total_extra_kwh += kwh
        start_hhmm, end_hhmm = window.split("-")
        scenario_descriptions.append(f"{label}: {'+' if kwh >= 0 else ''}{kwh} kWh/day ({window})")

        print()
        print("=" * 60)
        scenario_name = " + ".join(labels[:i + 1])
        print(f"SCENARIO: {scenario_name}")
        for desc in scenario_descriptions:
            print(f"  {desc}")
        print(f"  Total adjustment: {'+' if total_extra_kwh >= 0 else ''}{total_extra_kwh} kWh/day")
        print("=" * 60)

        args.extra_daily_kwh = total_extra_kwh
        args.extra_start = start_hhmm
        args.extra_end = end_hhmm

        scenario_dir = out_dir / f"scenario_{i + 1}"
        args.out_dir = str(scenario_dir)
        rc = run_analyse(args)
        if rc != 0:
            return rc

    # Print combined summary
    print()
    print("=" * 60)
    print("SUMMARY: All scenarios compared")
    print("=" * 60)

    all_rows = []
    for row in baseline_rows:
        row["scenario"] = "baseline"
        all_rows.append(row)

    cumulative_labels = []
    for i, label in enumerate(labels):
        cumulative_labels.append(label)
        scenario_dir = out_dir / f"scenario_{i + 1}"
        scenario_df = pd.read_csv(scenario_dir / "summary.csv")
        for row in scenario_df.to_dict(orient="records"):
            row["scenario"] = " + ".join(cumulative_labels)
            all_rows.append(row)

    headers = ["scenario", "tariff", "optimised_import_gbp", "optimised_export_gbp", "standing_charge_gbp", "optimised_net_gbp"]
    print_summary_table(all_rows, headers=headers)

    combined_df = pd.DataFrame(all_rows)
    write_csv(combined_df, out_dir / "model_summary.csv")
    print(f"\nModel summary written to: {out_dir / 'model_summary.csv'}")

    args.out_dir = str(out_dir)
    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    saved = load_defaults()

    def d(key, fallback):
        """Return saved default if available, otherwise the hardcoded fallback."""
        return saved.get(key, fallback)

    def _set_tariff_defaults(parser_obj):
        """Apply all tariff rate/standing-charge defaults to a subparser.

        Shared between the `default` and `model` subparsers; per-subparser
        keys (start_date, end_date, refresh_tariffs, tariffs, extra_*) stay inline.
        """
        parser_obj.set_defaults(
            intelligent_offpeak_import=d("intelligent_offpeak_import", 7.0),
            intelligent_peak_import=d("intelligent_peak_import", 26.0),
            intelligent_export=d("intelligent_export", 15.0),
            intelligent_standing_charge=d("intelligent_standing_charge", 57.01),
            intelligent_offpeak_start=d("intelligent_offpeak_start", "23:30"),
            intelligent_offpeak_end=d("intelligent_offpeak_end", "05:30"),
            agile_standing_charge=d("agile_standing_charge", 66.26),
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

    parser = argparse.ArgumentParser(
        description="Compare Tesla Powerwall usage against Octopus tariffs with optimal battery simulation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  # Save your settings once
  %(prog)s set-defaults --region-code M --battery-capacity-kwh 13 --email you@example.com

  # Download a year of Powerwall data (uses saved email)
  %(prog)s download-data

  # Run analysis with saved defaults
  %(prog)s default --power-csv download/power.csv

  # Override a saved default for one run
  %(prog)s default --power-csv power.csv --region-code C
""",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── set-defaults ──────────────────────────────────────────────
    sd = sub.add_parser("set-defaults",
        help="Save default settings so you don't have to type them every time")
    sd.add_argument("--region-code", help="Octopus region code (run list-regions to see options)")
    sd.add_argument("--battery-capacity-kwh", type=float, help="Powerwall usable capacity in kWh")
    sd.add_argument("--extra-daily-kwh", type=float, help="Extra daily consumption to model (kWh)")
    sd.add_argument("--extra-start", help="Start of extra consumption window (HH:MM)")
    sd.add_argument("--extra-end", help="End of extra consumption window (HH:MM)")
    sd.add_argument("--email", help="Tesla account email for download-data")
    sd.add_argument("--out-dir", help="Directory for output CSVs")
    sd.add_argument("--cache-dir", help="Directory for cached tariff CSVs")
    sd.add_argument("--intelligent-offpeak-import", type=float, help="Intelligent off-peak import rate (p/kWh)")
    sd.add_argument("--intelligent-peak-import", type=float, help="Intelligent peak import rate (p/kWh)")
    sd.add_argument("--intelligent-export", type=float, help="Intelligent export rate (p/kWh)")
    sd.add_argument("--intelligent-standing-charge", type=float, help="Intelligent daily standing charge (p/day)")
    sd.add_argument("--intelligent-offpeak-start", help="Intelligent off-peak window start (HH:MM)")
    sd.add_argument("--intelligent-offpeak-end", help="Intelligent off-peak window end (HH:MM)")
    sd.add_argument("--agile-standing-charge", type=float, help="Agile daily standing charge (p/day)")
    sd.add_argument("--show", action="store_true", help="Show current saved defaults and exit")

    # ── default ───────────────────────────────────────────────────
    default = sub.add_parser("default",
        help="Run full battery optimisation analysis across all tariffs",
        description="Run a full battery simulation for each tariff to find the cheapest option. "
                    "Automatically refreshes Powerwall data from Tesla if --email is saved.")
    default.add_argument("--power-csv", default="download/power.csv",
        help="Path to Powerwall power CSV (default: %(default)s)")
    default.add_argument("--email", default=d("email", None),
        help="Tesla email — if provided, auto-downloads latest data before analysis")
    default.add_argument("--out-dir", default=d("out_dir", "output"),
        help="Directory for output CSVs (default: %(default)s)")
    default.add_argument("--cache-dir", default=d("cache_dir", ".cache/tariffs"),
        help="Directory for cached tariff CSVs (default: %(default)s)")
    default.add_argument("--region-code", default=d("region_code", "M"),
        help="Octopus region code (default: %(default)s)")
    default.add_argument("--start-date", help="Only include data from this date onwards (YYYY-MM-DD)")
    default.add_argument("--end-date", help="Only include data up to this date (YYYY-MM-DD)")
    default.add_argument("--battery-capacity-kwh", type=float, default=d("battery_capacity_kwh", 13.0),
        help="Powerwall usable capacity in kWh (default: %(default)s)")
    default.add_argument("--battery-max-charge-kw", type=float, default=5.0,
        help="Maximum battery charge rate in kW (default: %(default)s)")
    default.add_argument("--battery-max-discharge-kw", type=float, default=5.0,
        help="Maximum battery discharge rate in kW (default: %(default)s)")
    default.add_argument("--battery-efficiency", type=float, default=0.90,
        help="Battery round-trip efficiency 0-1 (default: %(default)s)")
    default.add_argument("--refresh-tariffs", action="store_true",
        help="Re-download tariff data even if cached")
    default.add_argument("--tariffs",
        help="Comma-separated list of tariffs to simulate (default: all). "
             "Options: intelligent,agile,go,flux,cosy,flexible,tracker")
    default.set_defaults(
        extra_daily_kwh=d("extra_daily_kwh", 0.0),
        extra_start=d("extra_start", "17:00"),
        extra_end=d("extra_end", "22:00"),
    )
    _set_tariff_defaults(default)

    # ── download-data ─────────────────────────────────────────────
    download = sub.add_parser("download-data",
        help="Download a year of 5-minute Powerwall data from Tesla")
    download.add_argument("--email", default=d("email", None),
        help="Tesla account email address" + (f" (default: {d('email', None)})" if d("email", None) else ""))

    # ── list-regions ──────────────────────────────────────────────
    regions = sub.add_parser("list-regions",
        help="Show supported Octopus region codes and names")
    regions.add_argument("--json", action="store_true",
        help="Output as JSON instead of plain text")

    # ── refresh-tariffs ───────────────────────────────────────────
    refresh = sub.add_parser("refresh-tariffs",
        help="Re-download Agile tariff CSVs into cache")
    refresh.add_argument("--region-code", default=d("region_code", "M"),
        help="Octopus region code (default: %(default)s)")
    refresh.add_argument("--cache-dir", default=d("cache_dir", ".cache/tariffs"),
        help="Directory for cached tariff CSVs (default: %(default)s)")

    # ── model ─────────────────────────────────────────────────────
    model = sub.add_parser("model",
        help="Model a scenario with changed energy usage (e.g. new EV, hot tub)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""Model how a change in energy usage would affect your tariff costs.

Takes your real Powerwall data and applies adjustments to simulate scenarios
like buying an EV, adding a hot tub, or installing extra solar panels.""",
        epilog="""examples:
  # Model adding an EV that charges 7.5 kWh/day overnight
  %(prog)s model --power-csv power.csv --label "New EV" --adjust-kwh 7.5 --window 00:30-05:30

  # Model a hot tub using 5 kWh/day in the evening
  %(prog)s model --power-csv power.csv --label "Hot tub" --adjust-kwh 5 --window 17:00-22:00
""")
    model.add_argument("--power-csv", required=True, help="Path to Powerwall power CSV")
    model.add_argument("--label", action="append", dest="labels", required=True,
        help="Name for this scenario adjustment (repeat for multiple)")
    model.add_argument("--adjust-kwh", action="append", dest="adjust_kwhs", type=float, required=True,
        help="Daily kWh change: positive=more usage, negative=less (repeat for multiple)")
    model.add_argument("--window", action="append", dest="windows", required=True,
        help="Time window for the adjustment as HH:MM-HH:MM (repeat for multiple)")
    model.add_argument("--out-dir", default=d("out_dir", "output"),
        help="Directory for output CSVs (default: %(default)s)")
    model.add_argument("--cache-dir", default=d("cache_dir", ".cache/tariffs"),
        help="Directory for cached tariff CSVs (default: %(default)s)")
    model.add_argument("--region-code", default=d("region_code", "M"),
        help="Octopus region code (default: %(default)s)")
    model.add_argument("--battery-capacity-kwh", type=float, default=d("battery_capacity_kwh", 13.0),
        help="Powerwall usable capacity in kWh (default: %(default)s)")
    model.add_argument("--battery-max-charge-kw", type=float, default=5.0,
        help="Maximum battery charge rate in kW (default: %(default)s)")
    model.add_argument("--battery-max-discharge-kw", type=float, default=5.0,
        help="Maximum battery discharge rate in kW (default: %(default)s)")
    model.add_argument("--battery-efficiency", type=float, default=0.90,
        help="Battery round-trip efficiency 0-1 (default: %(default)s)")
    model.add_argument("--tariffs",
        help="Comma-separated list of tariffs to simulate (default: all). "
             "Options: intelligent,agile,go,flux,cosy,flexible,tracker")
    model.add_argument("--start-date", help="Only include data from this date onwards (YYYY-MM-DD)")
    model.add_argument("--end-date", help="Only include data up to this date (YYYY-MM-DD)")
    model.add_argument("--refresh-tariffs", action="store_true",
        help="Re-download tariff data even if cached")
    model.set_defaults(
        extra_daily_kwh=0.0,
        extra_start="17:00", extra_end="22:00",
        start_date=None, end_date=None, refresh_tariffs=False,
        tariffs=None,
    )
    _set_tariff_defaults(model)

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
    if args.command == "model":
        return run_model(args)
    if args.command == "default":
        # Auto-download latest data if email is available and using default power-csv path
        if getattr(args, "email", None):
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
        elif not Path(args.power_csv).exists():
            print(f"Error: {args.power_csv} not found.")
            print("Either:")
            print("  1. Save your email: set-defaults --email you@example.com")
            print("     Then run 'default' again to auto-download from Tesla")
            print("  2. Specify a CSV: default --power-csv /path/to/power.csv")
            return 1
        return run_analyse(args)
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


if __name__ == "__main__":
    raise SystemExit(main())
