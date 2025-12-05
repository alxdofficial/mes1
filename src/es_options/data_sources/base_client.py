"""Base data client interface for options data providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from es_options.utils.time_utils import DateLike


class BaseDataClient(ABC):
    """
    Abstract base class for options data clients.

    All data clients must implement these methods to provide a consistent
    interface for fetching options chains and underlying price data.
    """

    # Standard column names for options chain DataFrame
    CHAIN_COLUMNS = [
        "underlying_symbol",
        "option_symbol",
        "expiry",
        "strike",
        "right",  # 'C' or 'P'
        "bid",
        "ask",
        "last",
        "volume",
        "open_interest",
        "iv",  # implied volatility (optional)
        "delta",  # optional
        "gamma",  # optional
        "theta",  # optional
        "vega",  # optional
    ]

    # Standard column names for OHLC DataFrame
    OHLC_COLUMNS = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    @abstractmethod
    def get_option_chain(
        self,
        symbol: str,
        trade_date: DateLike,
        expiry: DateLike | None = None,
    ) -> pd.DataFrame:
        """
        Fetch options chain for a given symbol and date.

        Args:
            symbol: Underlying symbol (e.g., "ES", "SPX")
            trade_date: Trade date for which to fetch the chain
            expiry: Optional specific expiry date. If None, fetch all expiries.

        Returns:
            DataFrame with columns:
                - underlying_symbol: str
                - option_symbol: str
                - expiry: datetime
                - strike: float
                - right: str ('C' or 'P')
                - bid: float
                - ask: float
                - last: float
                - volume: int
                - open_interest: int
                - iv: float (optional)
                - delta, gamma, theta, vega: float (optional)
        """
        pass

    @abstractmethod
    def get_underlying_ohlc(
        self,
        symbol: str,
        start_date: DateLike,
        end_date: DateLike,
    ) -> pd.DataFrame:
        """
        Fetch OHLC data for the underlying instrument.

        Args:
            symbol: Underlying symbol (e.g., "ES", "SPX")
            start_date: Start date for historical data
            end_date: End date for historical data

        Returns:
            DataFrame with columns:
                - date: datetime (index)
                - open: float
                - high: float
                - low: float
                - close: float
                - volume: int
        """
        pass

    @abstractmethod
    def get_available_expiries(
        self,
        symbol: str,
        trade_date: DateLike,
    ) -> list[date]:
        """
        Get list of available expiry dates for a symbol.

        Args:
            symbol: Underlying symbol
            trade_date: Trade date

        Returns:
            List of available expiry dates
        """
        pass

    def validate_chain(self, df: pd.DataFrame) -> bool:
        """
        Validate that a chain DataFrame has required columns.

        Args:
            df: Options chain DataFrame

        Returns:
            True if valid, raises ValueError otherwise
        """
        required = ["expiry", "strike", "right", "bid", "ask"]
        missing = [col for col in required if col not in df.columns]

        if missing:
            raise ValueError(f"Chain missing required columns: {missing}")

        return True
