"""Pipeline for building RND surfaces from options chains."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from es_options.data_sources.base_client import BaseDataClient
from es_options.io.file_store import FileStore
from es_options.metrics.rnd_extractor import DataQuality, RNDExtractor, RNDResult
from es_options.processing.arbitrage_checks import ArbitrageChecker
from es_options.processing.chain_cleaner import ChainCleaner, ChainQualityMetrics
from es_options.processing.smoothing import CallPriceSmoother, SmoothingResult
from es_options.utils.time_utils import DateLike, parse_date


@dataclass
class RNDPipelineResult:
    """Result of RND pipeline execution."""
    symbol: str
    trade_date: date
    rnd_results: list[RNDResult]
    smoothing_results: dict[date, SmoothingResult]
    num_expiries: int
    success: bool
    errors: list[str]


class RNDPipeline:
    """
    Pipeline for RND extraction.

    Steps:
    1. Fetch options chain
    2. Clean and filter chain
    3. Smooth call prices for each expiry
    4. Check for arbitrage violations
    5. Extract RND using Breeden-Litzenberger
    """

    def __init__(
        self,
        client: BaseDataClient,
        file_store: FileStore | None = None,
        r: float = 0.05,
        min_dte: int = 30,
        max_dte: int = 180,
    ):
        self.client = client
        self.file_store = file_store or FileStore()
        self.r = r
        self.min_dte = min_dte
        self.max_dte = max_dte

        self.cleaner = ChainCleaner(min_dte=min_dte, max_dte=max_dte)
        self.smoother = CallPriceSmoother()
        self.arb_checker = ArbitrageChecker()
        self.rnd_extractor = RNDExtractor(r=r)

    def run(
        self,
        symbol: str,
        trade_date: DateLike,
        spot_price: float | None = None,
        save: bool = True,
    ) -> RNDPipelineResult:
        """Run the full RND pipeline."""
        trade_dt = parse_date(trade_date)
        errors = []

        # Fetch chain
        try:
            chain = self.client.get_option_chain(symbol, trade_dt)
            if chain.empty:
                return self._error_result(symbol, trade_dt, "Empty chain returned")
        except Exception as e:
            return self._error_result(symbol, trade_dt, f"Failed to fetch chain: {e}")

        # Get spot price
        if spot_price is None:
            spot_price = self.client.get_spot_price(symbol, trade_dt)
            if spot_price is None:
                spot_price = chain["strike"].median()

        # Clean chain
        clean_chain = self.cleaner.clean(chain, pd.Timestamp(trade_dt))

        if clean_chain.empty:
            return self._error_result(symbol, trade_dt, "No valid options after cleaning")

        # Build OTM chain (uses put-call parity for left tail)
        otm_chain, quality_metrics = self.cleaner.build_otm_chain(
            clean_chain, spot_price, r=self.r, return_quality=True
        )

        if otm_chain.empty:
            return self._error_result(symbol, trade_dt, "No valid OTM options")

        # Process each expiry
        smoothing_results = {}
        rnd_results = []

        for expiry in sorted(otm_chain["expiry"].unique()):
            expiry_date = expiry.date() if hasattr(expiry, "date") else expiry

            try:
                expiry_data = otm_chain[otm_chain["expiry"] == expiry]
                if len(expiry_data) < 10:
                    continue

                # Smooth - pass the OTM chain which already has 'mid' column
                smooth_result = self.smoother.fit(expiry_data, spot_price, price_col="mid")
                smoothing_results[expiry_date] = smooth_result

                # Check arbitrage
                strike_grid = smooth_result.get_strike_grid(200)
                arb_result = self.arb_checker.check(smooth_result.smooth_fn, strike_grid)

                if not arb_result.is_valid:
                    errors.append(f"{expiry_date}: {arb_result.num_violations} arb violations")

                # Extract RND with data quality info
                rnd = self.rnd_extractor.extract(smooth_result, expiry_date, trade_dt)

                # Attach data quality metrics
                if expiry_date in quality_metrics:
                    qm = quality_metrics[expiry_date]
                    rnd.data_quality = DataQuality(
                        num_raw_options=qm.num_raw,
                        num_otm_options=qm.num_otm,
                        num_calls=qm.num_calls,
                        num_puts_synthetic=qm.num_puts_synthetic,
                        strike_range=qm.strike_range,
                        strike_coverage=qm.strike_coverage,
                        avg_strike_gap=qm.avg_strike_gap,
                        max_strike_gap=qm.max_strike_gap,
                        moneyness_range=qm.moneyness_range,
                        quality_score=qm.quality_score,
                    )

                rnd_results.append(rnd)

            except Exception as e:
                errors.append(f"{expiry_date}: {e}")

        if not rnd_results:
            return self._error_result(symbol, trade_dt, "No RNDs extracted")

        # Save results
        if save and self.file_store:
            self.file_store.save_chain(clean_chain, symbol, trade_dt, "processed")

        return RNDPipelineResult(
            symbol=symbol,
            trade_date=trade_dt,
            rnd_results=rnd_results,
            smoothing_results=smoothing_results,
            num_expiries=len(rnd_results),
            success=True,
            errors=errors,
        )

    def _error_result(self, symbol: str, trade_date: date, error: str) -> RNDPipelineResult:
        return RNDPipelineResult(
            symbol=symbol,
            trade_date=trade_date,
            rnd_results=[],
            smoothing_results={},
            num_expiries=0,
            success=False,
            errors=[error],
        )
