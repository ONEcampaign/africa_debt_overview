"""Extract raw data"""

from bblocks.data_importers import InternationalDebtStatistics, get_dsa, WEO, GHED
from pydeflate import imf_exchange, set_pydeflate_path
import unesco_reader as uis

from scripts.config import Paths, START_YEAR, GHED_END_YEAR
from scripts.logger import logger
from scripts.utils import timeout_30min, add_africa_values


set_pydeflate_path(Paths.raw_data)


@timeout_30min
def get_debt_stocks_data() -> None:
    """Get the raw data for the International Debt Statistics."""

    logger.info("Downloading IDS debt stocks data...")

    ids = InternationalDebtStatistics()
    inds = list(ids.debt_stock_indicators.indicator_code.unique())

    df = ids.get_data(inds, include_labels=True, start_year=START_YEAR)

    # add Africa values
    df = add_africa_values(df, agg_operation="sum")

    df.to_parquet(Paths.raw_data / "ids_debt_stocks.parquet", index=False)

    logger.info("IDS debt stocks data downloaded successfully.")


@timeout_30min
def get_debt_service_data() -> None:
    """Get the raw data for the International Debt Statistics total debt service."""

    logger.info("Downloading IDS debt service data...")

    ids = InternationalDebtStatistics()
    inds = list(ids.debt_service_indicators.indicator_code.unique())

    df = ids.get_data(inds, include_labels=True, start_year=START_YEAR)

    # add Africa values
    df = add_africa_values(df, agg_operation="sum")

    df.to_parquet(Paths.raw_data / "ids_debt_service.parquet", index=False)

    logger.info("IDS debt service data downloaded successfully.")


def get_dsa_data() -> None:
    """Get the raw data for the IMF Debt Sustainability Analysis."""

    logger.info("Downloading DSA data...")

    df = get_dsa()
    df.to_csv(Paths.raw_data / "dsa.csv", index=False)

    logger.info("DSA data downloaded successfully.")


def get_gdp_data() -> None:
    """Get the raw data for GDP from WEO"""

    logger.info("Downloading GDP data...")

    weo = WEO()
    df = weo.get_data()

    df = (
        df.loc[lambda d: d.indicator_code == "NGDPD"]
        .assign(value=lambda d: d.value * d.scale_code)
        .loc[:, ["entity_code", "entity_name", "value", "year"]]
    )

    df.to_csv(Paths.raw_data / "gdp.csv", index=False)

    logger.info("GDP data downloaded successfully.")


def get_gov_expenditure_data() -> None:
    """Get the raw data for Government Expenditure from WEO"""

    logger.info("Downloading GOV expenditure data...")

    weo = WEO()
    df = (
        weo.get_data()
        .loc[lambda d: d.indicator_code == "GGX"]
        .assign(value=lambda d: d.value * d.scale_code)
        .loc[:, ["entity_code", "year", "value"]]
    )

    # TODO: Temp fix for pydeflate error, remove Hong Kong and Macao
    df = df.loc[lambda d: ~d.entity_code.isin(["HKG", "MAC", "CHN"])]

    # deflate
    df = df.pipe(
        imf_exchange,
        source_currency="LCU",
        target_currency="USD",
        id_column="entity_code",
        value_column="value",
        target_value_column="value",
    ).dropna(subset=["value"])

    df.to_csv(Paths.raw_data / "gov_expenditure.csv", index=False)

    logger.info("GOV expenditure data downloaded successfully.")


def get_health_expenditure_data() -> None:
    """Get the raw data for Health Expenditure from WEO"""

    logger.info("Downloading Health expenditure data...")

    ghed = GHED()
    df = ghed.get_data()

    df = (
        df.loc[
            lambda d: (d.indicator_code == "gghed_gge") & (d.year <= GHED_END_YEAR),
            ["iso3_code", "country_name", "value", "year"],
        ]
        .rename(columns={"iso3_code": "entity_code", "country_name": "entity_name"})
        .dropna(subset=["value"])
    )

    df.to_csv(Paths.raw_data / "health_expenditure.csv", index=False)

    logger.info("Health expenditure data downloaded successfully.")


def get_education_expenditure_data() -> None:
    """Get the raw data for Education Expenditure from WEO"""

    logger.info("Downloading Education expenditure data...")

    df = (
        uis.get_data("XGOVEXP.IMF", labels=True)
        .loc[:, ["geoUnit", "geoUnitName", "year", "value"]]
        .rename(
            columns={
                "geoUnit": "entity_code",
                "geoUnitName": "entity_name",
            }
        )
    )

    df.to_csv(Paths.raw_data / "education_expenditure.csv", index=False)

    logger.info("Education expenditure data downloaded successfully.")


if __name__ == "__main__":
    logger.info("Extracting raw data...")

    get_debt_stocks_data()  # debt stocks
    get_debt_service_data()  # debt service
    get_dsa_data()  # debt sustainability analysis
    get_gdp_data()  # gdp data
    get_gov_expenditure_data()  # government expenditure data
    get_health_expenditure_data()  # health expenditure data
    get_education_expenditure_data()  # education expenditure data

    logger.info("Raw data extraction complete.")
