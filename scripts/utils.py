"""Utility functions for various tasks."""

import signal
from collections.abc import Callable
from typing import Any
import pandas as pd
from bblocks import places


def timeout_30min(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to time out a function after 30 minutes and implement a
    try except block to catch any exceptions raised within the function."""

    def handler(signum: int, frame: Any) -> None:
        raise TimeoutError("Function timed out after 30 minutes")

    def wrapper() -> Any:
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(30 * 60)  # 30 minutes
        try:
            return func()
        except Exception as e:
            raise RuntimeError(f"Could not complete data download: {e!s}") from e
        finally:
            signal.alarm(0)

    return wrapper


def add_africa_values(df, agg_operation: str = "sum") -> pd.DataFrame:
    """Add Africa (excluding high income) aggregate values to a dataframe.

    Args:
        df: DataFrame containing country level data with columns 'entity_name' and 'value'.
        agg_operation: Aggregation operation to use when calculating the Africa values. Default is 'sum

    Returns:
        DataFrame with Africa (excluding high income) aggregate values added.
    """

    dff = df.copy(deep=True)  # make a copy to avoid modifying the original dataframe

    afr_dff = (
        df.assign(
            iso3_code=lambda d: places.resolve_places(
                d.entity_name, to_type="iso3_code", not_found="ignore"
            )
        )
        .dropna(subset=["iso3_code"])
        .assign(
            continent=lambda d: places.resolve_places(
                d.iso3_code, from_type="iso3_code", to_type="region"
            )
        )
        .assign(
            income_level=lambda d: places.resolve_places(
                d.iso3_code, from_type="iso3_code", to_type="income_level"
            )
        )
        .loc[lambda d: (d.continent == "Africa") & (d.income_level != "High income")]
        .drop(columns=["income_level", "iso3_code", "continent"])
        .dropna(subset="value")
        .groupby(
            [
                i
                for i in dff.columns
                if i not in ["value", "entity_name", "entity_code"]
            ],
            observed=True,
        )
        .agg({"value": agg_operation})
        .reset_index()
        .assign(entity_name="Africa (excluding high income)", is_aggregate=True)
    )

    dff = pd.concat([dff, afr_dff], ignore_index=True)

    return dff
