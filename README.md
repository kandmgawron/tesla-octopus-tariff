# Octopus Powerwall Tariff Compare

A Python CLI to compare historical Tesla Powerwall usage against Octopus tariffs using:

- Tesla 5-minute `power_all.csv` style exports
- saved Octopus Agile import / outgoing CSVs
- optional Tesla fetch mode via `teslapy`
- optional local Powerwall gateway capture via `pyPowerwall`

It can answer questions like:

- Was **Agile** or **Intelligent** cheaper over my actual history?
- What happens if I add **10 kWh/day** for a sauna or hot tub?
- If I add a **second 13 kWh battery**, how many days still spill onto the expensive rate?
- Which tariff is probably best for me in my region based on past usage?

## What it models

### Intelligent
- fixed cheap import window, for example `23:30–05:30`
- fixed peak import rate
- fixed export rate
- **daily cheap-energy cap = battery capacity**
- once the battery is empty, extra daily usage falls onto the expensive rate

### Agile
- actual half-hour import prices from saved CSVs
- actual half-hour outgoing prices from saved CSVs
- flexible demand is shifted into the **cheapest half-hours each day**
- configurable:
  - charge window hours per day
  - max kW charging rate
  - manual extra usage window and kWh/day

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Tariff file naming

The script expects files in a folder like:

```text
csv_agile_M_Yorkshire.csv
csv_agileoutgoing_M_Yorkshire.csv
```

Where:

- `M` is the Octopus region code
- `Yorkshire` is the region name

You can list what the script finds with:

```bash
python octopus_powerwall_tariff_compare.py discover-tariffs --tariff-dir .
```

## Basic compare example

```bash
python octopus_powerwall_tariff_compare.py compare \
  --power-csv power_all.csv \
  --tariff-dir . \
  --region-code M \
  --region-name Yorkshire \
  --battery-capacity-kwh 13 \
  --extra-daily-kwh 10 \
  --extra-start 17:00 \
  --extra-end 22:00 \
  --intelligent-offpeak-import 7 \
  --intelligent-peak-import 26 \
  --intelligent-export 15 \
  --intelligent-standing-charge 57.01 \
  --agile-standing-charge 66.26 \
  --agile-flex-hours 6 \
  --agile-flex-max-kw 3.3 \
  --out-dir output
```

That writes:

- `output/summary.csv`
- `output/daily_breakdown_intelligent.csv`
- `output/daily_breakdown_agile.csv`

## Current setup vs second battery

Current setup:

```bash
python octopus_powerwall_tariff_compare.py compare \
  --power-csv power_all.csv \
  --tariff-dir . \
  --region-code M \
  --region-name Yorkshire \
  --battery-capacity-kwh 13 \
  --extra-daily-kwh 10 \
  --intelligent-offpeak-import 7 \
  --intelligent-peak-import 26 \
  --intelligent-export 15 \
  --intelligent-standing-charge 57.01 \
  --agile-standing-charge 66.26
```

With a second 13 kWh battery:

```bash
python octopus_powerwall_tariff_compare.py compare \
  --power-csv power_all.csv \
  --tariff-dir . \
  --region-code M \
  --region-name Yorkshire \
  --battery-capacity-kwh 26 \
  --extra-daily-kwh 10 \
  --intelligent-offpeak-import 7 \
  --intelligent-peak-import 26 \
  --intelligent-export 15 \
  --intelligent-standing-charge 57.01 \
  --agile-standing-charge 66.26
```

## Tesla fetch mode

### TeslaPy
This is best-effort because Tesla's owner-facing API shapes can change.

```bash
python octopus_powerwall_tariff_compare.py fetch-tesla \
  --mode teslapy \
  --email you@example.com \
  --start-date 2025-01-01 \
  --end-date 2025-12-31 \
  --out-csv data/power_from_tesla.csv
```

### Local Powerwall gateway
This creates a fresh CSV by polling the gateway over time.

```bash
python octopus_powerwall_tariff_compare.py fetch-tesla \
  --mode local-gateway \
  --gateway-host 192.168.1.50 \
  --samples 288 \
  --poll-seconds 300 \
  --out-csv data/power_gateway_capture.csv
```

## Manual usage adjustments

The manual load adjustment is intentionally simple and transparent:

- `--extra-daily-kwh 10`
- `--extra-start 17:00`
- `--extra-end 22:00`

That spreads the extra 10 kWh/day evenly across the chosen half-hour window.

## Daily breakdown columns

### Intelligent daily CSV
- `existing_import_offpeak_kwh`
- `existing_import_peak_kwh`
- `export_kwh`
- `extra_kwh`
- `cheap_battery_kwh_available`
- `extra_cheap_kwh`
- `extra_expensive_kwh`
- `battery_ran_out`
- `import_cost_p`
- `export_revenue_p`
- `standing_charge_p`
- `net_cost_p`

### Agile daily CSV
- `direct_import_kwh`
- `export_kwh`
- `extra_kwh`
- `flexible_shifted_kwh`
- `stranded_in_original_window_kwh`
- `battery_ran_out`
- `import_cost_p`
- `export_revenue_p`
- `standing_charge_p`
- `net_cost_p`

## Notes

- This is a tariff comparison model, not a detailed electrochemical battery simulator.
- The Tesla fetch integrations are included because they are practical, but CSV mode is the most robust path for historical analysis.
- Standing charges are passed in as user inputs because tariff versions can vary.
