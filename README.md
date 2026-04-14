# Octopus Powerwall Tariff Compare

A simpler Python CLI to compare Tesla Powerwall usage against Octopus tariffs.

This version improves two things:

- it downloads the latest Agile and Agile Outgoing CSVs directly from the public Energy Stats CSV index
- it adds a `default` command so first-time use needs far fewer flags

The Energy Stats CSV index exposes region-coded `csv_agile_*` and `csv_agileoutgoing_*` files, so the script can fetch them automatically instead of asking users to download them first. citeturn227673view0

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Easiest first run

```bash
python octopus_powerwall_tariff_compare.py default --power-csv power_all.csv
```

Default assumptions:
- region `M` = Yorkshire
- Intelligent = 7p off-peak, 26p peak, 15p export
- Intelligent standing charge = 57.01p/day
- Agile standing charge = 66.26p/day
- battery = 13 kWh
- no extra daily load
- EV charging exclusion enabled
- output folder = `output`

## Common examples

Add an extra 10 kWh/day:

```bash
python octopus_powerwall_tariff_compare.py default --power-csv power_all.csv --extra-daily-kwh 10
```

Use a bigger battery:

```bash
python octopus_powerwall_tariff_compare.py default --power-csv power_all.csv --battery-capacity-kwh 26
```

Change region:

```bash
python octopus_powerwall_tariff_compare.py default --power-csv power_all.csv --region-code C
```

Refresh cached tariffs:

```bash
python octopus_powerwall_tariff_compare.py refresh-tariffs --region-code M
```

List region codes:

```bash
python octopus_powerwall_tariff_compare.py list-regions
```

## Advanced mode

```bash
python octopus_powerwall_tariff_compare.py compare \
  --power-csv power_all.csv \
  --region-code M \
  --extra-daily-kwh 10 \
  --battery-capacity-kwh 13
```

## Outputs

- `output/summary.csv`
- `output/daily_breakdown_intelligent.csv`
- `output/daily_breakdown_agile.csv`

## EV exclusion rule

By default, likely EV charging sessions are excluded from battery-backed cheap energy logic:
- around 6 kW+
- between 23:30 and 04:30

Those intervals are still costed as direct grid import.

Disable that rule with:

```bash
python octopus_powerwall_tariff_compare.py default --power-csv power_all.csv --no-ev-exclusion
```
