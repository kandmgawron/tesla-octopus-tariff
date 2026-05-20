# Octopus Powerwall Tariff Compare

A Python CLI that downloads your Tesla Powerwall usage data and simulates optimal battery behaviour across all major Octopus Energy tariffs to find the absolute cheapest one for you.

## How It Works

Unlike simple rate calculators, this tool runs a **full battery optimisation simulation** for each tariff. For every half-hour slot in your data, it decides the optimal action:

- **Solar decision**: If the export rate exceeds the import rate, export ALL solar and buy from grid (net profit). Otherwise, solar serves home load first.
- **Battery charging**: Charges from grid during the cheapest import slots each day.
- **Battery discharging**: Discharges to avoid expensive grid imports, or exports to grid when export rates are high.
- **Efficiency losses**: Accounts for 90% round-trip battery efficiency (configurable).
- **Physical limits**: Respects battery capacity, max charge/discharge rates, and 10% reserve floor.
- **Realistic state**: Battery state of charge carries across days — no artificial daily reset. If the battery is depleted one evening, it stays low until the next cheap charging window.

## Supported Tariffs

| Tariff | Type | Description |
|--------|------|-------------|
| **Intelligent** | Time-of-use | Cheap overnight rate (23:30–05:30) for smart EV/battery owners |
| **Agile** | Half-hourly variable | Prices change every 30 minutes based on wholesale market |
| **Go** | Time-of-use | Simple cheap overnight rate for EV owners |
| **Flux** | 3-band import+export | Designed for solar+battery with peak export premium |
| **Cosy** | 3 cheap windows | Designed for heat pumps (04:00–07:00, 13:00–16:00, 22:00–00:00) |
| **Tracker** | Daily variable | Single rate per day tied to wholesale prices |
| **Flexible (SVR)** | Flat rate | Standard variable rate — the default tariff |

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

### 1. Save your defaults (one-time setup)

```bash
python octopus_powerwall_tariff_compare.py set-defaults \
  --email your-tesla-email@example.com \
  --region-code M
```

### 2. Run the analysis

```bash
python octopus_powerwall_tariff_compare.py
```

That's it. This auto-downloads your Powerwall data from Tesla (up to a year of history), fetches tariff rates, then simulates optimal battery behaviour for all 7 tariffs and shows which is cheapest. Per-day data is cached in `download/power/` so subsequent runs only download new days.

To compare only specific tariffs:

```bash
python octopus_powerwall_tariff_compare.py --tariffs flux,intelligent,agile
```

To also refresh cached tariff rate data:

```bash
python octopus_powerwall_tariff_compare.py --refresh-tariffs
```

**`--power-csv`** — You only need this if your Powerwall data lives somewhere other than the default `download/power.csv` (e.g. you exported it manually or have multiple sites). If you've used `set-defaults --email` the tool downloads and merges data to that path automatically.

## Commands

| Command | Description |
|---------|-------------|
| _(none)_ / `default` | Run full battery optimisation analysis across all tariffs (auto-downloads Powerwall data when `--email` is saved) |
| `model` | Model scenarios like adding an EV or hot tub |
| `download-data` | Download 5-minute Powerwall data from Tesla (up to 1 year) |
| `set-defaults` | Save settings (region, battery, email) so you don't retype them |
| `list-regions` | Show supported Octopus region codes |
| `refresh-tariffs` | Re-download Agile tariff CSVs |

## Scenario Modelling

Model how changes to your energy usage would affect costs:

```bash
# What if I buy an EV that charges 7.5 kWh/day overnight?
python octopus_powerwall_tariff_compare.py model \
  --power-csv download/power.csv \
  --label "New EV" --adjust-kwh 7.5 --window 00:30-05:30

# What about an EV AND a hot tub (5 kWh/day evenings)?
python octopus_powerwall_tariff_compare.py model \
  --power-csv download/power.csv \
  --label "EV" --adjust-kwh 7.5 --window 00:30-05:30 \
  --label "Hot tub" --adjust-kwh 5 --window 17:00-22:00

# Model reduced usage (e.g. better insulation saving 3 kWh/day)
python octopus_powerwall_tariff_compare.py model \
  --power-csv download/power.csv \
  --label "Insulation" --adjust-kwh -3 --window 06:00-22:00
```

## Common Options

```bash
# Use a different region (C = London)
python octopus_powerwall_tariff_compare.py --power-csv power.csv --region-code C

# Bigger battery (e.g. 2x Powerwall 2)
python octopus_powerwall_tariff_compare.py --power-csv power.csv --battery-capacity-kwh 26

# Powerwall 3 (higher charge/discharge rate)
python octopus_powerwall_tariff_compare.py --power-csv power.csv \
    --battery-max-charge-kw 11.5 --battery-max-discharge-kw 11.5

# Account for battery degradation
python octopus_powerwall_tariff_compare.py --power-csv power.csv --battery-efficiency 0.85

# View saved defaults
python octopus_powerwall_tariff_compare.py set-defaults --show

# List all region codes
python octopus_powerwall_tariff_compare.py list-regions
```

## Outputs

Results are written to the `output/` directory:

- `summary.csv` — All tariffs ranked by optimised net cost
- `daily_<tariff>.csv` — Daily breakdown for each tariff
- `model_summary.csv` — Combined scenario comparison (when using `model`)

## Tesla Login

The `download-data` command (and the auto-download triggered by `default --email`) uses [TeslaPy](https://github.com/tdorssers/TeslaPy) to authenticate. On first run:

1. A URL is printed — open it in your browser
2. Log in to your Tesla account
3. You'll see a "Verified Successfully" page (it will then hang — this is expected)
4. Open your browser's Developer Tools (Cmd+Option+I on Mac)
5. Go to the Console tab
6. Find the message: `Failed to launch 'tesla://auth/callback?code=...'`
7. Right-click the `tesla://auth/callback?...` URL and choose "Copy link address"
8. Paste the full URL into the terminal

Your token is cached locally so you won't need to log in again unless it expires.

## Using With Other Batteries

This tool works with any home battery system, not just Tesla Powerwalls. You just need your usage data in the right CSV format.

**Battery specs are auto-detected** from the data — capacity, max charge rate, and max discharge rate are all inferred from observed behaviour. You can override any of them manually if needed:

```bash
python octopus_powerwall_tariff_compare.py --power-csv my_data.csv \
    --battery-capacity-kwh 10 \
    --battery-max-charge-kw 3.6 \
    --battery-max-discharge-kw 3.6
```

### Required CSV Format

Your CSV needs 5-minute interval readings with these columns:

| Column | Required | Description |
|--------|----------|-------------|
| `timestamp` | Yes | ISO 8601 timestamp with timezone (e.g. `2025-09-01T14:00:00+01:00`) |
| `grid_power` | Yes | Grid power in watts. Positive = importing, negative = exporting |
| `solar_power` | No | Solar generation in watts (0 if no solar) |
| `battery_power` | No | Battery power in watts. Negative = charging, positive = discharging |
| `load_power` | No | Home consumption in watts |

Example rows:

```csv
timestamp,solar_power,battery_power,grid_power,load_power
2025-09-01T00:00:00+01:00,0.0,-5000.0,5300.0,300.0
2025-09-01T00:05:00+01:00,0.0,-5000.0,5600.0,600.0
2025-09-01T12:00:00+01:00,3500.0,0.0,-3200.0,300.0
2025-09-01T18:00:00+01:00,0.0,2000.0,-1800.0,200.0
```

**Minimum required**: `timestamp` and `grid_power`. Without `solar_power` and `battery_power`, the tool can still run the simulation but won't auto-detect battery specs and will assume no solar generation.

**Interval**: Data should be at 5-minute intervals. The tool aggregates to 30-minute slots internally.

### Getting Data From Other Systems

- **GivEnergy**: Export from the GivEnergy portal or use the GivTCP API
- **SolarEdge**: Export from the monitoring portal (may need to convert from 15-min to 5-min)
- **Solis/Ginlong**: Export from SolisCloud
- **Home Assistant**: Export energy sensor data with a 5-minute recording interval
- **Enphase**: Export from the Enlighten portal

As long as you can produce a CSV with `timestamp` and `grid_power` at 5-minute intervals, the tool will work.
