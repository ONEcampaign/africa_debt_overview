# African debt overview

African debt overview page

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install dependencies
uv sync
```


## Usage

Scripts are located in the `scripts/` directory.

```bash
# Run an analysis script
uv run python scripts/example_analysis.py
```

### Project Structure

- `scripts/`: Analysis scripts
  - `config.py`: Project paths configuration
  - `logger.py`: Logging setup
- `raw_data/`: Input data files
- `output/`: Analysis results and outputs

### Code Quality

```bash
# Run linter
uv run ruff check scripts/

# Run formatter
uv run ruff format scripts/

# Run type checker
uv run mypy scripts/
```


## Pre-commit Hooks

Pre-commit hooks are configured to run automatically. To manually run:

```bash
pre-commit run --all-files
```


## License

This project is licensed under the mit License - see the LICENSE file for details.

