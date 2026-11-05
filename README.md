# African Debt Overview

Data pipeline and analysis behind the [African Debt Overview](https://data.one.org/analysis/african-debt) page.
It fetches debt data from international sources (World Bank IDS, IMF WEO, IMF DSA, GHED, UNESCO), 
processes it, and outputs chart-ready JSON/CSV files and downloadable datasets.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
```

## Usage

```bash
# 1. Download raw data
uv run python scripts/extract_raw_data.py

# 2. Run analysis and generate outputs
uv run python scripts/analysis.py
```

Outputs are saved to `output/` as JSON and CSV files (one per chart, plus a `key_stats.json` summary).

Raw data in `raw_data/` is not tracked in version control, except for the IMF list of
debt sustainability assessments (`raw_data/imf_dsa.csv`). The IMF currently blocks
scripted downloads of that list. When the download fails, `extract_raw_data.py` logs a
warning and the analysis uses the saved copy. To update it by hand, save
https://www.imf.org/external/Pubs/ft/dsa/DSAlist.pdf from a browser and run:

```bash
uv run python -m scripts.update_dsa path/to/DSAlist.pdf
```

## Project Structure

- `scripts/` — Analysis and data extraction scripts
  - `extract_raw_data.py` — Fetches data from external APIs
  - `analysis.py` — Processes data and generates charts/downloads
  - `update_dsa.py` — Updates the saved IMF DSA list from a PDF downloaded by hand
  - `utils.py` — Helper functions
  - `config.py` — Paths, year ranges, and region settings
  - `logger.py` — Logging setup
- `raw_data/` — Cached input data
- `output/` — Generated chart data and downloadable datasets

## Code Quality

```bash
uv run ruff check scripts/
uv run ruff format scripts/
uv run mypy scripts/
```

Pre-commit hooks are configured to run automatically. To run manually:

```bash
pre-commit run --all-files
```

## License

This project is licensed under the MIT License — see the LICENSE file for details.
