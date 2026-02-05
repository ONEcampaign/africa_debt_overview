"""Module to run analysis and generate charts"""

import pandas as pd
import numpy as np
from bblocks import places

from scripts.utils import filter_african_debtors, custom_sort, format_values
from scripts.config import Paths, LATEST_YEAR, START_YEAR, NUM_EST_YEARS, SORT_PARAMS
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
        # remove white space from all columns
        .assign(
            debtor_name=lambda d: d.debtor_name.str.strip(),
            creditor_name=lambda d: d.creditor_name.str.strip(),
            category=lambda d: d.category.str.strip(),
        )
        .pipe(custom_sort, SORT_PARAMS)
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
    chart_data = (
        dff.pivot(
            index=["debtor_name", "year", "creditor_name"],
            columns="category",
            values="value",
        )
        .reset_index()
        .pipe(custom_sort, SORT_PARAMS)
    )

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


def _get_gdp_df() -> pd.DataFrame:
    """Read and prepare GDP data for analysis."""

    # read gdp data
    gdp = pd.read_csv(Paths.raw_data / "gdp.csv")

    return (
        gdp.assign(
            iso3_code=lambda d: places.resolve_places(
                d.entity_code, to_type="iso3_code", not_found="ignore"
            )
        )
        .dropna(subset=["iso3_code"])
        .loc[:, ["iso3_code", "year", "value"]]
        .rename(columns={"value": "gdp"})
    )


def chart_2() -> None:
    """Chart 2: line, debt stocks as a percent of GDP"""

    # read debt stocks data and gdp data
    df = pd.read_parquet(Paths.raw_data / "ids_debt_stocks.parquet")
    gdp = _get_gdp_df()

    # prepare debt stocks data
    df = (
        df.pipe(_prepare_debt_stocks_data)
        .loc[lambda d: d.creditor_name == "All creditors"]  # only total debt
        .groupby(["debtor_name", "year"], observed=True)  # aggregate across categories
        .agg({"value": "sum"})
        .reset_index()
        .assign(
            iso3_code=lambda d: places.resolve_places(
                d.debtor_name, to_type="iso3_code", not_found="ignore"
            )
        )
        .dropna(subset=["iso3_code"])  # keep only countries
    )

    # merge with gdp data and calculate debt to gdp ratio
    df = (
        df.merge(gdp, on=["iso3_code", "year"], how="left", validate="many_to_one")
        .dropna(subset=["gdp"])  # drop rows with missing gdp data
        .assign(debt_to_gdp=lambda d: d.value / d.gdp * 100)
        .drop(columns=["value", "gdp"])  # drop unnecessary columns
    )

    # calculate Africa median debt stocks to gdp ratio
    africa_median = (
        df.groupby(["year"], observed=True)
        .agg({"debt_to_gdp": "median"})
        .assign(debtor_name="Africa (excluding high income) (median)")
        .reset_index()
    )

    # combine country data with Africa median
    df = (
        pd.concat([df, africa_median], ignore_index=True)
        .rename(columns={"debt_to_gdp": "value"})
        .pipe(custom_sort, {"debtor_name": ["Africa (excluding high income) (median)"]})
    )

    # export download data
    df.to_csv(Paths.output / "chart_2_download.csv", index=False)

    # prepare chart data and export
    df = df.pivot(index="year", columns="debtor_name", values="value").reset_index()
    df.to_csv(Paths.output / "chart_2_chart.csv", index=False)

    logger.info("Chart 2 generated successfully.")


def chart_3() -> None:
    """Chart 3: treemap, debt stocks  by creditor and creditor type"""

    # read debt stocks data
    df = pd.read_parquet(Paths.raw_data / "ids_debt_stocks.parquet")

    # prepare data
    df = (
        df.pipe(_prepare_debt_stocks_data)
        .loc[lambda d: d.year == LATEST_YEAR,]
        .loc[
            lambda d: d.creditor_name != "All creditors"
        ]  # exclude "All creditors" for this view
    )

    # export download data
    df.to_csv(Paths.output / "chart_3_download.csv", index=False)

    # prepare chart data and export
    df = df.assign(value_annotation=lambda d: d.value.apply(format_values)).assign(
        name_annotation=lambda d: np.where(
            d.category.isin(["bilateral", "commercial banks", "other private"]),
            d["creditor_name"].astype(str) + " (" + d["category"].astype(str) + ")",
            d["creditor_name"].astype(str),
        )
    )

    df.to_csv(Paths.output / "chart_3_chart.csv", index=False)

    logger.info("Chart 3 generated successfully.")


def chart_4() -> None:
    """Chart 4: bar chart, china proportion lending"""

    # read debt stocks data
    df = pd.read_parquet(Paths.raw_data / "ids_debt_stocks.parquet")

    # prepare data
    df = df.pipe(_prepare_debt_stocks_data)

    # create a China df
    china_df = (
        df.loc[lambda d: d.creditor_name == "China"]
        # replace commercial banks and other private with private
        .assign(
            category=lambda d: d.category.replace(
                {"commercial banks": "private", "other private": "private"}
            )
        )
        .groupby(["debtor_name", "year", "category"], observed=True)
        .agg({"value": "sum"})
        .reset_index()
        .assign(category=lambda d: "China " + "(" + d.category + ")")
    )

    # create a other creditors excluding china df
    other_df = (
        df.loc[lambda d: ~d.creditor_name.isin(["All creditors", "China"])]
        .groupby(["debtor_name", "year"], observed=True)
        .agg({"value": "sum"})
        .reset_index()
        .assign(category="Other creditors")
    )

    # combine both dataframes
    combined_df = pd.concat([china_df, other_df], ignore_index=True).rename(
        columns={"category": "creditor"}
    )

    # export download data
    combined_df.to_csv(Paths.output / "chart_4_download.csv", index=False)

    # prepare chart data and export
    chart_df = (
        combined_df.pivot(
            index=["debtor_name", "year"], columns="creditor", values="value"
        )
        .reset_index()
        .pipe(custom_sort, {"debtor_name": SORT_PARAMS["debtor_name"]})
    )

    chart_df.to_csv(Paths.output / "chart_4_chart.csv", index=False)

    logger.info("Chart 4 generated successfully.")


def _prepare_debt_service_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare debt service data for analysis."""

    mapping = {
        "DT.AMT.PBND.CD": {"category": "bonds", "type": "principal"},
        "DT.AMT.BLAT.CD": {"category": "bilateral", "type": "principal"},
        "DT.AMT.PCBK.CD": {"category": "commercial banks", "type": "principal"},
        "DT.AMT.MLAT.CD": {"category": "multilateral", "type": "principal"},
        "DT.AMT.PROP.CD": {"category": "other private", "type": "principal"},
        "DT.INT.BLAT.CD": {"category": "bilateral", "type": "interest"},
        "DT.INT.MLAT.CD": {"category": "multilateral", "type": "interest"},
        "DT.INT.PBND.CD": {"category": "bonds", "type": "interest"},
        "DT.INT.PCBK.CD": {"category": "commercial banks", "type": "interest"},
        "DT.INT.PROP.CD": {"category": "other private", "type": "interest"},
    }

    # filter African debtors
    df = filter_african_debtors(df)

    return (
        df.dropna(subset="value")
        .loc[
            lambda d: (d.year >= START_YEAR) & (d.year <= LATEST_YEAR + NUM_EST_YEARS),
            [
                "indicator_name",
                "indicator_code",
                "year",
                "entity_name",
                "counterpart_name",
                "value",
            ],
        ]
        .assign(
            counterpart_name=lambda d: d.counterpart_name.replace(
                {"World": "All creditors"}
            )
        )
        .rename(
            columns={"entity_name": "debtor_name", "counterpart_name": "creditor_name"}
        )
        .assign(
            category=lambda d: d.indicator_code.map(lambda x: mapping[x]["category"]),
            type=lambda d: d.indicator_code.map(lambda x: mapping[x]["type"]),
        )
        .drop(columns="indicator_code")
        .assign(
            debtor_name=lambda d: d.debtor_name.str.strip(),
            creditor_name=lambda d: d.creditor_name.str.strip(),
        )
        .reset_index(drop=True)
    )


def chart_5() -> None:
    """Chart 5: line chart, debt service payments over time"""

    # read debt service data
    df = pd.read_parquet(Paths.raw_data / "ids_debt_service.parquet")

    # prepare data
    df = _prepare_debt_service_data(df)

    # remove debtor/creditor pairs where all values are zero
    df = (
        df.groupby(["debtor_name", "creditor_name"], observed=True)
        .filter(lambda d: d["value"].sum() != 0)
        .reset_index(drop=True)
    )

    # aggregate across categories to get total debt service payments by debtor and creditor
    df = (
        df.groupby(["year", "debtor_name", "creditor_name", "category"], observed=True)
        .agg({"value": "sum"})
        .reset_index()
    )

    # sort and export download data
    df.pipe(custom_sort, SORT_PARAMS).to_csv(
        Paths.output / "chart_5_download.csv", index=False
    )

    # prepare chart data and export
    chart_data = (
        df.pivot(
            index=["debtor_name", "year", "creditor_name"],
            columns="category",
            values="value",
        )
        .reset_index()
        .pipe(custom_sort, SORT_PARAMS)
    )

    chart_data.to_csv(Paths.output / "chart_5_chart.csv", index=False)

    # chart json data
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
            Paths.output / "chart_5_chart.json", orient="records", date_format="iso"
        )
    )

    logger.info("Chart 5 generated successfully.")


def _calculate_debt_service_pct_gov_spending(ds_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate debt service as a percent of government spending.

    Args:
        ds_df: formatted debt service dataframe
    """

    gov_df = pd.read_csv(Paths.raw_data / "gov_expenditure.csv")

    return (ds_df
     .merge(gov_df.rename(columns={"value": "gov_spending"}),
            how="left",
            on=["entity_code", "year"],
            validate="one_to_one")
     .assign(ds_pct_gov_spending=lambda d: d["value"] / d["gov_spending"] * 100)
     .drop(columns="gov_spending")

     )

def chart_6() -> None:
    """Chart 6: line chart, debt service percent of government spending with
    health and education expenditure
    """

    # get debt service data as a percent of government spending
    ds_df = pd.read_parquet(Paths.raw_data / "ids_debt_service.parquet")
    ds_df = _prepare_debt_service_data(ds_df)
    ds_df = (ds_df
             .loc[lambda d: (d.creditor_name == "All creditors") & (d.year <= LATEST_YEAR)]
             .groupby(["debtor_name", "year"], as_index=False)
             .agg({"value": "sum"})
             .assign(entity_code=lambda d: places.resolve_places(d.debtor_name,
                                                                 to_type="iso3_code",
                                                                 not_found="ignore"))
             .dropna(subset="entity_code")
             .reset_index(drop=True)
             )
    ds_df = _calculate_debt_service_pct_gov_spending(ds_df)

    # merge health expenditure data
    health_df = pd.read_csv(Paths.raw_data / "health_expenditure.csv")
    ds_df = (ds_df
             .merge((health_df
                     .loc[:, ["entity_code", "year", "value"]]
                     .rename(columns={"value": "health_expenditure"})
                     ),
                    how="left",
                    on=["entity_code", "year"],
                    validate="one_to_one"
                    )
             )

    # merge education expenditure data
    education_df = pd.read_csv(Paths.raw_data / "education_expenditure.csv")
    ds_df = (ds_df.merge((education_df
               .loc[:, ["entity_code", "year", "value"]]
               .rename(columns={"value": "education_expenditure"})
               ),
              how="left",
              on=["entity_code", "year"],
              validate="one_to_one"
              )
     )

    # Add Africa median values
    africa_median = (
        ds_df.assign(
            region=lambda d: places.resolve_places(
                d.entity_code, from_type="iso3_code", to_type="region"
            )
        )
        .loc[lambda d: d.region == "Africa"]
        .groupby("year", as_index=False)
        .agg({"ds_pct_gov_spending": "median",
              "health_expenditure": "median",
              "education_expenditure": "median"})
        .assign(debtor_name="Africa (excluding high income) (median)")
    )

    ds_df = pd.concat([ds_df, africa_median], ignore_index=True)

    # final formatting
    ds_df = (ds_df
    .drop(columns="value")
             .rename(columns={"ds_pct_gov_spending": "debt service",
                       "health_expenditure": "health expenditure",
                       "education_expenditure": "education expenditure"
                       })
             .pipe(custom_sort, {"debtor_name": SORT_PARAMS["debtor_name"]})
             )

    # export download data
    ds_df.to_csv(Paths.output / "chart_6_download.csv", index=False)

    #export chart data
    ds_df.to_csv(Paths.output / "chart_6_chart.csv", index=False)

    logger.info("Chart 6 generated successfully.")


if __name__ == "__main__":
    logger.info("Generating charts...")

    chart_1()  # Chart 1: Bar, Total debt stocks
    chart_2()  # Chart 2: line, debt stocks as a percent of GDP
    chart_3()  # Chart 3: treemap, debt stocks  by creditor and creditor type
    chart_4()  # Chart 4: bar chart, china proportion lending
    chart_5()  # Chart 5: line chart, debt service payments over time
    chart_6()  # Chart 6: line chart, debt service percent of government spending with health and education expenditure

    logger.info("All charts generated successfully.")
