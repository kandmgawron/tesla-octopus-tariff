# Octopus Powerwall Tariff Compare

A Python CLI that downloads your Tesla Powerwall usage data and compares Octopus Intelligent vs Agile tariffs to show which is cheaper for you.

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

This downloads a full year of 5-minute power data from Tesla. It will prompt you to log in via your browser on first run.

```bash
python octopus_powerwall_tariff_compare.py download-data
```

Data is saved to `download/<site_id>/power.csv`. Per-day files are cached in `download/<site_id>/power/` so subsequent runs only download new days.

### 3. Run the tariff comparison

```bash
python octopus_powerwall_tariff_compare.py default \
  --power-csv download/<site_id>/power.csv
```

Replace `<site_id>` with your Tesla energy site ID (printed during download).

### All-in-one: download + refresh tariffs + compare

```bash
python octopus_powerwall_tariff_compare.py full-refresh
```

This does steps 2 and 3 together, plus refreshes the Agile tariff data.

## Commands

| Command | Description |
|---------|-------------|
| `set-defaults` | Save settings (region, battery, email) so you don't retype them |
| `download-data` | Download 5-minute Powerwall data from Tesla (up to 1 year) |
| `full-refresh` | Download data + refresh tariffs + run comparison in one go |
| `default` | Run comparison with sensible defaults |
| `compare` | Run comparison with full control over every tariff parameter |
| `model` | Model scenarios like adding an EV or hot tub |
| `list-regions` | Show supported Octopus region codes |
| `refresh-tariffs` | Re-download Agile tariff CSVs |

## Scenario Modelling

Model how changes to your energy usage would affect costs:

```bash
# What if I buy an EV that charges 7.5 kWh/day overnight?
python octopus_powerwall_tariff_compare.py model \
  --power-csv download/<site_id>/power.csv \
  --label "New EV" --adjust-kwh 7.5 --window 00:30-05:30

# What about an EV AND a hot tub (5 kWh/day evenings)?
python octopus_powerwall_tariff_compare.py model \
  --power-csv download/<site_id>/power.csv \
  --label "EV" --adjust-kwh 7.5 --window 00:30-05:30 \
  --label "Hot tub" --adjust-kwh 5 --window 17:00-22:00

# Model reduced usage (e.g. better insulation saving 3 kWh/day)
python octopus_powerwall_tariff_compare.py model \
  --power-csv download/<site_id>/power.csv \
  --label "Insulation" --adjust-kwh -3 --window 06:00-22:00
```

## Common Options

```bash
# Use a different region (C = London)
python octopus_powerwall_tariff_compare.py default \
  --power-csv power.csv --region-code C

# Bigger battery (e.g. 2x Powerwall 2)
python octopus_powerwall_tariff_compare.py default \
  --power-csv power.csv --battery-capacity-kwh 26

# Add extra daily load (e.g. 10 kWh for heat pump)
python octopus_powerwall_tariff_compare.py default \
  --power-csv power.csv --extra-daily-kwh 10

# Disable EV charging detection
python octopus_powerwall_tariff_compare.py default \
  --power-csv power.csv --no-ev-exclusion

# View saved defaults
python octopus_powerwall_tariff_compare.py set-defaults --show

# List all region codes
python octopus_powerwall_tariff_compare.py list-regions
```

## Outputs

Results are written to the `output/` directory:

- `summary.csv` - Side-by-side tariff comparison
- `daily_breakdown_intelligent.csv` - Daily costs on Intelligent
- `daily_breakdown_agile.csv` - Daily costs on Agile
- `model_summary.csv` - Combined scenario comparison (when using `model`)

## Default Assumptions

| Setting | Default | Description |
|---------|---------|-------------|
| Region | M (Yorkshire) | Octopus pricing region |
| Battery | 13 kWh | Single Powerwall 2 usable capacity |
| Intelligent off-peak | 7p/kWh | Import rate 23:30-05:30 |
| Intelligent peak | 26p/kWh | Import rate outside off-peak |
| Intelligent export | 15p/kWh | Export rate |
| Intelligent standing | 57.01p/day | Daily standing charge |
| Agile standing | 66.26p/day | Daily standing charge |
| EV detection | Enabled | Excludes >5.5kW draws between 23:30-04:30 |

All defaults can be overridden via `set-defaults` or on the command line.

## Tesla Login

The `download-data` and `full-refresh` commands use the [TeslaPy](https://github.com/tdorssers/TeslaPy) library to authenticate with Tesla. On first run:

1. A URL is printed - open it in your browser
2. Log in to your Tesla account
3. You'll get a "Page Not Found" error - this is expected
4. Copy the full URL from your browser's address bar and paste it back

Your token is cached locally so you won't need to log in again unless it expires.
