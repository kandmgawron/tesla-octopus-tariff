# Octopus Powerwall Tariff Compare

A Python CLI that downloads your Tesla Powerwall usage data and simulates optimal battery behaviour across all major Octopus Energy tariffs to find the absolute cheapest one for you.

## How It Works

Unlike simple rate calculators, this tool runs a **full battery optimisation simulation** for each tariff. For every half-hour slot in your data, it decides the optimal action:

- **Solar decision**: If the export rate exceeds the import rate, export ALL solar and buy from grid (net profit). Otherwise, solar serves home load first.
- **Battery charging**: Charges from grid during the cheapest import slots each day.
- **Battery discharging**: Discharges to avoid expensive grid imports, or exports to grid when export rates are high.
- **Efficiency losses**: Accounts for 90% round-trip battery efficiency (configurable).
- **Physical limits**: Respects battery capacity and max charge/discharge rates.

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
  --region-code M \
  --battery-capacity-kwh 13
```

### 2. Download your Powerwall data

```bash
python octopus_powerwall_tariff_compare.py download-data
```

Data is saved to `download/power.csv`. Per-day files are cached in `download/power/`.

### 3. Run the analysis

```bash
python octopus_powerwall_tariff_compare.py default
```

This auto-downloads any new Powerwall data, then simulates optimal battery behaviour for all 7 tariffs and shows which is cheapest.

To compare only specific tariffs:

```bash
python octopus_powerwall_tariff_compare.py default --power-csv download/power.csv --tariffs flux,intelligent,agile
```

### All-in-one: download + refresh tariffs + analyse

```bash
python octopus_powerwall_tariff_compare.py full-refresh
```

## Commands

| Command | Description |
|---------|-------------|
| `default` | Run full battery optimisation analysis across all tariffs |
| `model` | Model scenarios like adding an EV or hot tub |
| `download-data` | Download 5-minute Powerwall data from Tesla (up to 1 year) |
| `full-refresh` | Download data + refresh tariffs + run analysis in one go |
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
python octopus_powerwall_tariff_compare.py default --power-csv power.csv --region-code C

# Bigger battery (e.g. 2x Powerwall 2)
python octopus_powerwall_tariff_compare.py default --power-csv power.csv --battery-capacity-kwh 26

# Powerwall 3 (higher charge/discharge rate)
python octopus_powerwall_tariff_compare.py default --power-csv power.csv \
    --battery-max-charge-kw 11.5 --battery-max-discharge-kw 11.5

# Account for battery degradation
python octopus_powerwall_tariff_compare.py default --power-csv power.csv --battery-efficiency 0.85

# Disable EV charging detection
python octopus_powerwall_tariff_compare.py default --power-csv power.csv --no-ev-exclusion

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

The `download-data` and `full-refresh` commands use [TeslaPy](https://github.com/tdorssers/TeslaPy) to authenticate. On first run:

1. A URL is printed — open it in your browser
2. Log in to your Tesla account
3. You'll see a "Verified Successfully" page (it will then hang — this is expected)
4. Open your browser's Developer Tools (Cmd+Option+I on Mac)
5. Go to the Console tab
6. Find the message: `Failed to launch 'tesla://auth/callback?code=...'`
7. Right-click the `tesla://auth/callback?...` URL and choose "Copy link address"
8. Paste the full URL into the terminal

Your token is cached locally so you won't need to log in again unless it expires.
