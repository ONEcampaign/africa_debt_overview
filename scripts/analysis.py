"""Module to run analysis and generate charts"""

import pandas as pd

from scripts.utils import filter_african_debtors, custom_sort
from scripts.config import Paths
from scripts.logger import logger


def _prepare_debt_stocks_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare debt stocks data for analysis.

    - Filters for African debtor countries.
    - Maps indicator codes to debt categories.
    - Selects relevant columns and renames counterpart "World" to "All creditors".
    - Sorts data with custom order for entities and counterparts.
    """

    categories = {
        "DT.DOD.BLAT.CD": "bilateral",
        "DT.DOD.MLAT.CD": "multilateral",
        "DT.DOD.PBND.CD": "bonds",
        "DT.DOD.PCBK.CD": "commercial banks",
        "DT.DOD.PROP.CD": "other private",
    }

    # filter African debtors
    dff = filter_african_debtors(df)

    dff = (
        dff.dropna(subset=["value"])
        .assign(category=lambda d: d.indicator_code.map(categories))
        .loc[:, ["entity_name", "year", "category", "value", "counterpart_name"]]
        .reset_index(drop=True)
        .assign(
            counterpart_name=lambda d: d.counterpart_name.replace(
                {"World": "All creditors"}
            )
        )
        .rename(
            columns={"entity_name": "debtor_name", "counterpart_name": "creditor_name"}
        )
        .pipe(
            custom_sort,
            {
                "debtor_name": [
                    "Africa (excluding high income)",
                    "Sub-Saharan Africa (excluding high income)",
                ],
                "creditor_name": ["All creditors"],
            },
        )
    )

    return dff


def chart_1() -> None:
    """Chart 1: Bar, Total debt stocks"""

    # read debt stocks data
    debt_stocks = pd.read_parquet(Paths.raw_data / "ids_debt_stocks.parquet")

    # prepare data
    dff = _prepare_debt_stocks_data(debt_stocks)

    # save download data
    dff.to_csv(Paths.output / "chart_1_download.csv", index=False)

    # prepare chart data
    chart_data = dff.pivot(
        index=["debtor_name", "year", "creditor_name"],
        columns="category",
        values="value",
    ).reset_index()

    chart_data.to_csv(Paths.output / "chart_1_chart.csv", index=False)

    # generate chart json
    (
        chart_data.rename(
            columns={
                "debtor_name": "filter1_values",
                "year": "x_values",
                "creditor_name": "filter2_values",
                "bilateral": "y1",
                "multilateral": "y2",
                "bonds": "y3",
                "commercial banks": "y4",
                "other private": "y5",
            }
        )
        .assign(y_values=lambda d: d[["y1", "y2", "y3", "y4", "y5"]].values.tolist())
        .loc[:, ["filter1_values", "x_values", "filter2_values", "y_values"]]
        .to_json(
            Paths.output / "chart_1_chart.json", orient="records", date_format="iso"
        )
    )

    logger.info("Chart 1 generated successfully.")


if __name__ == "__main__":
    logger.info("Generating charts...")

    chart_1()  # Chart 1: Bar, Total debt stocks
