# Octopus Powerwall Tariff Compare

A Python CLI that downloads your Tesla Powerwall usage data and compares all major Octopus Energy tariffs to show which is cheapest for you.

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

This downloads a full year of 5-minute power data from Tesla. It will prompt you to log in via your browser on first run.

```bash
python octopus_powerwall_tariff_compare.py download-data
```

Data is saved to `download/power.csv`. Per-day files are cached in `download/power/` so subsequent runs only download new days.

### 3. Run the tariff comparison

```bash
python octopus_powerwall_tariff_compare.py default
```

This automatically refreshes your Powerwall data (downloads any new days), then compares all 7 Octopus tariffs. If no data exists yet, it downloads everything and tells you how long it will take (~10 minutes per year).

You can also skip the auto-download and just compare existing data:

```bash
python octopus_powerwall_tariff_compare.py compare --power-csv download/power.csv
```

To compare only specific tariffs:

```bash
# Just compare Go vs Intelligent
python octopus_powerwall_tariff_compare.py compare --power-csv download/power.csv --tariffs intelligent,go

# All variable tariffs
python octopus_powerwall_tariff_compare.py compare --power-csv download/power.csv --tariffs agile,tracker
```

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
| `best-tariff` | Full simulation: find the absolute best tariff with optimal battery use |
| `optimise-charging` | Show savings from moving battery charging to cheapest slots |
| `model` | Model scenarios like adding an EV or hot tub |
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

## Optimised Battery Charging

See how much you could save by scheduling your Powerwall to charge at the cheapest times:

```bash
# Analyse all tariffs
python octopus_powerwall_tariff_compare.py optimise-charging

# Just check variable tariffs where smart scheduling matters most
python octopus_powerwall_tariff_compare.py optimise-charging --tariffs agile,tracker,flux

# With a Powerwall 3 (higher charge rate = more flexibility)
python octopus_powerwall_tariff_compare.py optimise-charging --battery-max-charge-kw 11.5
```

This analyses your actual battery charging history — when the Powerwall charged from the grid and at what rate — then simulates moving that same energy to the cheapest available slots each day. It shows:

- **Actual cost**: what you paid for battery charging based on when it happened
- **Optimised cost**: what you would have paid if charging was scheduled optimally
- **Saving**: the difference, in £ and %

For time-of-use tariffs (Intelligent, Go), if your battery already charges overnight you'll see 0% saving — it's already optimal. For variable tariffs (Agile, Tracker) and multi-band tariffs (Flux, Cosy), smart scheduling can yield significant savings.

## Best Tariff (Full Simulation)

The most comprehensive analysis — simulates optimal battery behaviour for each tariff to find the absolute best one for your home:

```bash
# Find the best tariff with full battery optimisation
python octopus_powerwall_tariff_compare.py best-tariff

# Compare specific tariffs
python octopus_powerwall_tariff_compare.py best-tariff --tariffs flux,intelligent,agile

# With Powerwall 3 specs (higher charge/discharge rate)
python octopus_powerwall_tariff_compare.py best-tariff \
    --battery-max-charge-kw 11.5 --battery-max-discharge-kw 11.5

# Account for battery degradation
python octopus_powerwall_tariff_compare.py best-tariff --battery-efficiency 0.85
```

Unlike the basic `compare` command which just applies rates to your actual usage, `best-tariff` simulates what would happen with **perfect battery scheduling**:

- Charges from grid during the cheapest import slots
- Uses solar to serve home load first, then charges battery, then exports surplus
- Discharges battery to avoid importing during expensive periods
- Exports stored energy during peak export windows (e.g. Flux pays 28.60p/kWh 16:00–19:00)
- Accounts for battery round-trip efficiency losses (default 90%)
- Respects battery capacity and charge/discharge rate limits

This shows the true potential of each tariff if you had a smart controller (like Intelligent Octopus or Flux automation) optimally managing your battery.

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

- `summary.csv` - Side-by-side tariff comparison (all tariffs ranked by net cost)
- `daily_breakdown_intelligent.csv` - Daily costs on Intelligent
- `daily_breakdown_agile.csv` - Daily costs on Agile
- `daily_breakdown_go.csv` - Daily costs on Go
- `daily_breakdown_flux.csv` - Daily costs on Flux
- `daily_breakdown_cosy.csv` - Daily costs on Cosy
- `daily_breakdown_flexible.csv` - Daily costs on Flexible (SVR)
- `daily_breakdown_tracker.csv` - Daily costs on Tracker
- `model_summary.csv` - Combined scenario comparison (when using `model`)

## Default Assumptions

| Setting | Default | Description |
|---------|---------|-------------|
| Region | M (Yorkshire) | Octopus pricing region |
| Battery | 13 kWh | Single Powerwall 2 usable capacity |
| **Intelligent** | | |
| Off-peak rate | 7p/kWh | Import rate 23:30–05:30 |
| Peak rate | 26p/kWh | Import rate outside off-peak |
| Export rate | 15p/kWh | Export rate |
| Standing charge | 57.01p/day | Daily standing charge |
| **Go** | | |
| Off-peak rate | 8p/kWh | Import rate 23:30–05:30 |
| Peak rate | 24.5p/kWh | Import rate outside off-peak |
| Export rate | 12p/kWh | Export rate |
| Standing charge | 53.35p/day | Daily standing charge |
| **Flux** | | |
| Off-peak import | 9.80p/kWh | 02:00–05:00 |
| Day import | 22.36p/kWh | Outside peak and off-peak |
| Peak import | 33.54p/kWh | 16:00–19:00 |
| Off-peak export | 4.05p/kWh | 02:00–05:00 |
| Day export | 14.40p/kWh | Outside peak and off-peak |
| Peak export | 28.60p/kWh | 16:00–19:00 |
| Standing charge | 48.93p/day | Daily standing charge |
| **Cosy** | | |
| Cosy rate | 12p/kWh | 04:00–07:00, 13:00–16:00, 22:00–00:00 |
| Day rate | 24.50p/kWh | Standard hours |
| Peak rate | 36.75p/kWh | 16:00–19:00 |
| Export rate | 12p/kWh | Export rate |
| Standing charge | 53.35p/day | Daily standing charge |
| **Tracker** | | |
| Import rate | Daily variable | Fetched from Octopus API |
| Export rate | 12p/kWh | Export rate |
| Standing charge | 53.35p/day | Daily standing charge |
| **Flexible (SVR)** | | |
| Import rate | 24.50p/kWh | Flat rate |
| Export rate | 12p/kWh | Export rate |
| Standing charge | 53.35p/day | Daily standing charge |
| **Agile** | | |
| Standing charge | 66.26p/day | Daily standing charge |
| EV detection | Enabled | Excludes >5.5kW draws between 23:30–04:30 |

All defaults can be overridden via `set-defaults` or on the command line.

## Tesla Login

The `download-data` and `full-refresh` commands use the [TeslaPy](https://github.com/tdorssers/TeslaPy) library to authenticate with Tesla. On first run:

1. A URL is printed - open it in your browser
2. Log in to your Tesla account
3. You'll see a **"Verified Successfully"** page - the page will then hang (this is expected)
4. Open your browser's **Developer Tools** (Cmd+Option+I on Mac, F12 on Windows)
5. Go to the **Console** tab
6. Find the message: `Failed to launch 'tesla://auth/callback?code=...'`
7. Right-click the `tesla://auth/callback?...` URL and choose **Copy link address**
8. Paste the full URL into the terminal

Your token is cached locally so you won't need to log in again unless it expires.
