"""Update the saved IMF DSA list from a PDF downloaded by hand.

Use this when extract_raw_data can't download the list. Open
https://www.imf.org/external/Pubs/ft/dsa/DSAlist.pdf in a browser, save it, then run:

    uv run python -m scripts.update_dsa path/to/DSAlist.pdf

This parses the PDF the same way as extract_raw_data does after downloading it.
"""

import sys
from pathlib import Path

from scripts.config import Paths
from scripts.extract_raw_data import DSA_PATH, parse_dsa_pdf
from scripts.logger import logger


def update_dsa_from_pdf(pdf_path: Path) -> None:
    """Parse a DSA list PDF and save it as the repo's copy of the DSA data."""

    df = parse_dsa_pdf(pdf_path.read_bytes())
    df.to_csv(DSA_PATH, index=False)

    logger.info(
        f"Saved {len(df)} countries from {pdf_path.name} to "
        f"{DSA_PATH.relative_to(Paths.project)}"
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python -m scripts.update_dsa path/to/DSAlist.pdf")

    update_dsa_from_pdf(Path(sys.argv[1]).expanduser())
