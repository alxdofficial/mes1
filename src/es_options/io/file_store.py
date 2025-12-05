"""File storage for options data."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd

from es_options.config import get_config
from es_options.utils.time_utils import DateLike, parse_date


class FileStore:
    """File-based storage for options chains and computed data."""

    def __init__(
        self,
        base_path: Path | str | None = None,
        format: Literal["parquet", "csv"] = "parquet",
    ):
        if base_path is None:
            config = get_config()
            base_path = config.project_root / "data"

        self.base_path = Path(base_path)
        self.format = format

        for subdir in ["raw", "processed"]:
            (self.base_path / subdir).mkdir(parents=True, exist_ok=True)

    def _get_path(
        self,
        category: str,
        symbol: str,
        data_type: str,
        trade_date: date,
        stage: Literal["raw", "processed"] = "processed",
    ) -> Path:
        ext = "parquet" if self.format == "parquet" else "csv"
        date_str = trade_date.strftime("%Y%m%d")
        path = self.base_path / stage / category / symbol
        path.mkdir(parents=True, exist_ok=True)
        return path / f"{data_type}_{date_str}.{ext}"

    def save_chain(
        self,
        df: pd.DataFrame,
        symbol: str,
        trade_date: DateLike,
        stage: Literal["raw", "processed"] = "raw",
    ) -> Path:
        path = self._get_path("chains", symbol, "options_chain", parse_date(trade_date), stage)
        self._save_dataframe(df, path)
        return path

    def load_chain(
        self,
        symbol: str,
        trade_date: DateLike,
        stage: Literal["raw", "processed"] = "raw",
    ) -> pd.DataFrame | None:
        path = self._get_path("chains", symbol, "options_chain", parse_date(trade_date), stage)
        return self._load_dataframe(path)

    def _save_dataframe(self, df: pd.DataFrame, path: Path) -> None:
        if self.format == "parquet":
            df.to_parquet(path, index=True)
        else:
            df.to_csv(path, index=True)

    def _load_dataframe(self, path: Path) -> pd.DataFrame | None:
        if not path.exists():
            return None
        try:
            if self.format == "parquet":
                return pd.read_parquet(path)
            else:
                return pd.read_csv(path, index_col=0, parse_dates=True)
        except Exception:
            return None
