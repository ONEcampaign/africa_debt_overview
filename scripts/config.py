"""Project configuration and paths."""

from pathlib import Path


START_YEAR = 2000
LATEST_YEAR = 2024
NUM_EST_YEARS = 6
GHED_END_YEAR = 2023


class Paths:
    """Class to store the paths to the data and output folders."""

    project = Path(__file__).resolve().parent.parent
    raw_data = project / "raw_data"
    output = project / "output"
    scripts = project / "scripts"


# Ensure directories exist
Paths.raw_data.mkdir(exist_ok=True)
Paths.output.mkdir(exist_ok=True)
