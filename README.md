# Octopus Tariff Compare

Find the cheapest Octopus Energy tariff for your home battery and solar setup. Works with any battery system — Tesla Powerwall, SolarEdge, GivEnergy, Solis, or any other system that can provide usage data.

> **Not comfortable with Python?** Use the [web app](https://octopus-tariff-compare-web.vercel.app) instead — just upload your CSV and get results in your browser.

## How It Works

This tool runs a **full battery optimisation simulation** for each tariff using your real energy data. For every half-hour slot, it decides the optimal action:

- **Solar decision**: If the export rate exceeds the cost of stored energy, export all solar and serve load from battery. Otherwise, solar serves home load first.
- **Battery charging**: Charges from grid during the cheapest import slots each day.
- **Battery discharging**: Discharges to avoid expensive grid imports, or exports to grid when export rates are high.
- **Negative rates**: When import rates go negative (Agile), maximises grid consumption.
- **Physical limits**: Respects battery capacity, charge/discharge rates, grid export limit (DNO), and 10% reserve floor.
- **Realistic state**: Battery state of charge carries across days. If depleted one evening, it stays low until the next cheap charging window.
- **Auto-detection**: Battery capacity, charge/discharge rates, and grid export limit are all detected from your data automatically.

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
  --postcode "SW1A 1AA"
```

### 2. Get your energy data

Choose one of:

**Tesla Powerwall** — auto-downloads via API:
```bash
python octopus_powerwall_tariff_compare.py set-defaults --email your-tesla-email@example.com
python octopus_powerwall_tariff_compare.py download-data
```

**SolarEdge** — auto-downloads via API:
```bash
python octopus_powerwall_tariff_compare.py set-defaults \
  --solaredge-api-key YOUR_KEY --solaredge-site-id YOUR_ID
python octopus_powerwall_tariff_compare.py download-solaredge
```

**Any other system** — export a CSV manually (see [Data Format](#data-format) below) and place it at `download/power.csv`.

### 3. Run the analysis

```bash
python octopus_powerwall_tariff_compare.py
```

That's it. The tool simulates optimal battery behaviour for all 7 tariffs and shows which is cheapest for you annually.

To compare only specific tariffs:

```bash
python octopus_powerwall_tariff_compare.py --tariffs flux,intelligent,agile
```

## Commands

| Command | Description |
|---------|-------------|
| _(none)_ / `default` | Run full battery optimisation analysis across all tariffs |
| `model` | Model scenarios like adding an EV or hot tub |
| `download-data` | Download data from Tesla (up to 1 year) |
| `download-solaredge` | Download data from SolarEdge (up to 1 year) |
| `set-defaults` | Save settings so you don't retype them |
| `list-regions` | Show supported Octopus region codes |
| `refresh-tariffs` | Re-download Agile/Tracker rate data |

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
# Use a different region (by postcode)
python octopus_powerwall_tariff_compare.py --postcode "EC1A 1BB"

# Or by region code directly
python octopus_powerwall_tariff_compare.py --region-code C

# Override auto-detected battery specs
python octopus_powerwall_tariff_compare.py --battery-capacity-kwh 10 \
    --battery-max-charge-kw 3.6 --battery-max-discharge-kw 3.6

# Override grid export limit (e.g. G99 approval at 6 kW)
python octopus_powerwall_tariff_compare.py --grid-export-limit-kw 6.0

# Account for battery age (reduces capacity and efficiency)
python octopus_powerwall_tariff_compare.py --battery-age-years 5

# View saved defaults
python octopus_powerwall_tariff_compare.py set-defaults --show

# List all region codes
python octopus_powerwall_tariff_compare.py list-regions
```

## Outputs

Results are written to the `output/` directory:

- `summary.csv` — All tariffs ranked by annual cost
- `daily_<tariff>.csv` — Daily breakdown for each tariff
- `model_summary.csv` — Combined scenario comparison (when using `model`)

## Data Format

The tool works with any battery system as long as you can provide a CSV with energy data at 5-minute intervals.

### Required Columns

| Column | Required | Description |
|--------|----------|-------------|
| `timestamp` | Yes | ISO 8601 with timezone (e.g. `2025-09-01T14:00:00+01:00`) |
| `grid_power` | Yes | Grid power in watts. Positive = importing, negative = exporting |
| `solar_power` | No | Solar generation in watts |
| `battery_power` | No | Battery power in watts. Negative = charging, positive = discharging |
| `load_power` | No | Home consumption in watts |

### Example

```csv
timestamp,solar_power,battery_power,grid_power,load_power
2025-09-01T00:00:00+01:00,0.0,-5000.0,5300.0,300.0
2025-09-01T00:05:00+01:00,0.0,-5000.0,5600.0,600.0
2025-09-01T12:00:00+01:00,3500.0,0.0,-3200.0,300.0
2025-09-01T18:00:00+01:00,0.0,2000.0,-1800.0,200.0
```

**Minimum**: `timestamp` and `grid_power`. Without `solar_power` and `battery_power`, the tool still runs but won't auto-detect battery specs and assumes no solar.

**Interval**: 5-minute readings (aggregated to 30-minute slots internally). 15-minute data also works (e.g. SolarEdge).

### Getting Data From Other Systems

| System | How to get data |
|--------|----------------|
| **Tesla Powerwall** | Built-in: `download-data` command |
| **SolarEdge** | Built-in: `download-solaredge` command |
| **GivEnergy** | Export from portal or use GivTCP API |
| **Solis/Ginlong** | Export from SolisCloud |
| **Home Assistant** | Export energy sensors at 5-min interval |
| **Enphase** | Export from Enlighten portal |

## Tesla Login

The `download-data` command uses [TeslaPy](https://github.com/tdorssers/TeslaPy) to authenticate. On first run:

1. A URL is printed — open it in your browser
2. Log in to your Tesla account
3. You'll see a "Verified Successfully" page (it will then hang — this is expected)
4. Open Developer Tools (Cmd+Option+I on Mac), go to Console tab
5. Find `Failed to launch 'tesla://auth/callback?code=...'`
6. Right-click the URL and choose "Copy link address"
7. Paste into the terminal

Your token is cached locally so you won't need to log in again unless it expires.

## SolarEdge Setup

To get your API key: log in to the [SolarEdge monitoring portal](https://monitoring.solaredge.com), go to Admin > Site Access > API Access, and generate a key. Your site ID is shown in the portal URL.
