"""EOD Historical Data client for ES options chains."""

from __future__ import annotations

import time
from datetime import date
from typing import Any

import pandas as pd
import requests

from es_options.config import get_config
from es_options.utils.time_utils import DateLike, parse_date

from .base_client import BaseDataClient


class EODClient(BaseDataClient):
    """Client for EOD Historical Data API."""

    BASE_URL = "https://eodhistoricaldata.com/api"

    # Map symbols to EOD exchange suffixes
    SYMBOL_MAP = {
        "SPY": "SPY.US",
        "QQQ": "QQQ.US",
        "IWM": "IWM.US",
        "ES": "ES.CME",  # Note: ES futures options not available on EOD
    }

    def __init__(self, api_key: str | None = None):
        config = get_config()
        self._api_key = api_key or config.require_env("EOD_API_KEY")
        self._session = requests.Session()
        self._last_request_time = 0.0

    def _get_eod_symbol(self, symbol: str) -> str:
        """Map symbol to EOD format."""
        return self.SYMBOL_MAP.get(symbol, f"{symbol}.US")

    def _rate_limit_wait(self) -> None:
        """Enforce rate limiting (1 req/sec)."""
        elapsed = time.time() - self._last_request_time
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        self._last_request_time = time.time()

    def _make_request(self, endpoint: str, params: dict[str, Any] | None = None) -> dict | list:
        self._rate_limit_wait()

        url = f"{self.BASE_URL}{endpoint}"
        params = params or {}
        params["api_token"] = self._api_key
        params["fmt"] = "json"

        response = self._session.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    def get_option_chain(
        self,
        symbol: str,
        trade_date: DateLike,
        expiry: DateLike | None = None,
    ) -> pd.DataFrame:
        """Fetch options chain."""
        trade_dt = parse_date(trade_date)
        eod_symbol = self._get_eod_symbol(symbol)

        endpoint = f"/options/{eod_symbol}"
        params = {"trade_date": trade_dt.strftime("%Y-%m-%d")}

        if expiry:
            params["contract_name"] = parse_date(expiry).strftime("%Y-%m-%d")

        try:
            response = self._make_request(endpoint, params)
        except Exception:
            return pd.DataFrame(columns=self.CHAIN_COLUMNS)

        return self._parse_chain_response(response, symbol)

    def _parse_chain_response(self, response: dict | list, symbol: str) -> pd.DataFrame:
        if not response:
            return pd.DataFrame(columns=self.CHAIN_COLUMNS)

        # EOD returns: {"data": [{"expirationDate": ..., "options": {"CALL": [...], "PUT": [...]}}]}
        data = response.get("data", response) if isinstance(response, dict) else response
        records = []

        if isinstance(data, list):
            for expiry_data in data:
                if not isinstance(expiry_data, dict):
                    continue

                expiry_str = expiry_data.get("expirationDate", "")
                options = expiry_data.get("options", {})

                for option_type in ["CALL", "PUT"]:
                    option_list = options.get(option_type, [])
                    if not isinstance(option_list, list):
                        continue

                    for opt in option_list:
                        record = self._parse_option_record(opt, symbol, expiry_str)
                        if record:
                            records.append(record)

        if not records:
            return pd.DataFrame(columns=self.CHAIN_COLUMNS)

        df = pd.DataFrame(records)
        df["expiry"] = pd.to_datetime(df["expiry"])
        df["strike"] = df["strike"].astype(float)
        df["bid"] = pd.to_numeric(df["bid"], errors="coerce").fillna(0)
        df["ask"] = pd.to_numeric(df["ask"], errors="coerce").fillna(0)
        df["last"] = pd.to_numeric(df["last"], errors="coerce").fillna(0)
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
        df["open_interest"] = pd.to_numeric(df["open_interest"], errors="coerce").fillna(0).astype(int)

        return df.sort_values(["expiry", "strike", "right"]).reset_index(drop=True)

    def _parse_option_record(self, data: dict, symbol: str, expiry_str: str) -> dict | None:
        try:
            return {
                "underlying_symbol": symbol,
                "option_symbol": data.get("contractName", ""),
                "expiry": expiry_str,
                "strike": float(data.get("strike", 0)),
                "right": "C" if data.get("type") == "CALL" else "P",
                "bid": data.get("bid", 0),
                "ask": data.get("ask", 0),
                "last": data.get("lastPrice", 0),
                "volume": data.get("volume", 0),
                "open_interest": data.get("openInterest", 0),
                "iv": data.get("impliedVolatility"),
                "delta": data.get("delta"),
                "gamma": data.get("gamma"),
                "theta": data.get("theta"),
                "vega": data.get("vega"),
            }
        except (ValueError, TypeError):
            return None

    def get_underlying_ohlc(
        self, symbol: str, start_date: DateLike, end_date: DateLike
    ) -> pd.DataFrame:
        """Fetch OHLC data."""
        start_dt = parse_date(start_date)
        end_dt = parse_date(end_date)
        eod_symbol = self._get_eod_symbol(symbol)

        endpoint = f"/eod/{eod_symbol}"
        params = {
            "from": start_dt.strftime("%Y-%m-%d"),
            "to": end_dt.strftime("%Y-%m-%d"),
            "period": "d",
        }

        try:
            response = self._make_request(endpoint, params)
        except Exception:
            return pd.DataFrame(columns=self.OHLC_COLUMNS)

        if not response:
            return pd.DataFrame(columns=self.OHLC_COLUMNS)

        df = pd.DataFrame(response)
        df = df.rename(columns={"adjusted_close": "adj_close"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        return df

    def get_spot_price(self, symbol: str, trade_date: DateLike) -> float | None:
        """Get spot price for a symbol."""
        trade_dt = parse_date(trade_date)
        ohlc = self.get_underlying_ohlc(symbol, trade_dt, trade_dt)
        if ohlc.empty:
            return None
        return float(ohlc["close"].iloc[-1])

    def get_available_expiries(self, symbol: str, trade_date: DateLike) -> list[date]:
        """Get list of available expiry dates."""
        chain = self.get_option_chain(symbol, trade_date)
        if chain.empty:
            return []
        expiries = chain["expiry"].unique()
        return sorted([e.date() if hasattr(e, "date") else e for e in expiries])
