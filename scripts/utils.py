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


def filter_african_debtors(
    df: pd.DataFrame, debtor_col: str = "entity_name"
) -> pd.DataFrame:
    """Filter the DataFrame to include only African debtor countries and regions.

    Args:
        df: DataFrame containing debtor data with a column for debtor names.
        debtor_col: Name of the column in df that contains debtor names. Default is 'entity

    Returns:
        Filtered DataFrame containing only African debtor countries and regions.
    """

    groups = [
        "Sub-Saharan Africa (excluding high income)",
        "Africa (excluding high income)",
    ]

    filter_values = groups + places.filter_african_countries(
        list(df.entity_name.unique()), not_found="ignore"
    )

    return df.loc[lambda d: d[debtor_col].isin(filter_values)].reset_index(drop=True)


def custom_sort(
    df: pd.DataFrame, resort_dict: dict[str, list[str] | str]
) -> pd.DataFrame:
    """Custom sort columns placing specific items on top and sorting the
    rest alphabetically

    Args:
        df: DataFrame to resort
        resort_dict: Dictionary of columns and items to sort at the top of the column. The dictionary
           should be in the format {col name: items to sort at top}.

    Returns:
        A resorted DataFrame
    """

    # create a copy of df
    _df = df.copy(deep=True)

    # convert all values to list
    for k, v in resort_dict.items():
        if isinstance(v, str):
            resort_dict[k] = [resort_dict[k]]

        # check if all passed columns exist
        if k not in list(_df.columns):
            raise ValueError(f"Column not found: {k}")

    for col, values in resort_dict.items():
        _df[col] = pd.Categorical(
            _df[col],
            categories=values
            + sorted(val for val in _df[col].unique() if val not in values),
            ordered=True,
        )

    _df = _df.sort_values(list(resort_dict.keys()))
    return _df

def format_values(value: float | int) -> str:
    """Format a numeric value into a human-readable string with appropriate
    units (millions, billions, trillions).

    Args:
        value: Numeric value to format.

    Returns:
        Formatted string representation of the value.
    """
    if value < 1_000_000:
        return f"US${value:,.0f}"

    elif value < 1_000_000_000:
        scaled = value / 1_000_000
        fmt = f"{scaled:.1f}" if not scaled.is_integer() else f"{int(scaled)}"
        return f"US${fmt} million"

    elif value < 1_000_000_000_000:
        scaled = value / 1_000_000_000
        fmt = f"{scaled:.1f}" if not scaled.is_integer() else f"{int(scaled)}"
        return f"US${fmt} billion"

    else:
        scaled = value / 1_000_000_000_000
        fmt = f"{scaled:.1f}" if not scaled.is_integer() else f"{int(scaled)}"
        return f"US${fmt} trillion"
