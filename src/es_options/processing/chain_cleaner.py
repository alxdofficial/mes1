"""Options chain cleaning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ChainQualityMetrics:
    """Metrics about the option chain data quality."""
    num_raw: int
    num_after_clean: int
    num_otm: int
    num_calls: int
    num_puts_synthetic: int
    strikes: np.ndarray
    spot: float

    @property
    def strike_range(self) -> tuple[float, float]:
        return (float(self.strikes.min()), float(self.strikes.max()))

    @property
    def strike_coverage(self) -> float:
        """Strike range as fraction of spot price."""
        return (self.strikes.max() - self.strikes.min()) / self.spot

    @property
    def avg_strike_gap(self) -> float:
        if len(self.strikes) < 2:
            return 0.0
        return float(np.diff(np.sort(self.strikes)).mean())

    @property
    def max_strike_gap(self) -> float:
        if len(self.strikes) < 2:
            return 0.0
        return float(np.diff(np.sort(self.strikes)).max())

    @property
    def moneyness_range(self) -> tuple[float, float]:
        return (float(self.strikes.min() / self.spot), float(self.strikes.max() / self.spot))

    @property
    def quality_score(self) -> float:
        """
        Compute overall quality score (0-1).

        Factors:
        - Number of strikes (more = better, target 30+)
        - Strike coverage (wider = better, target 50%+ of spot)
        - Max gap (smaller = better, target < 10)
        - Balance (calls vs puts, target ~50/50)
        """
        # Number of strikes: 0-1 score, saturates at 30
        n_score = min(self.num_otm / 30, 1.0)

        # Coverage: 0-1 score, saturates at 60% of spot
        cov_score = min(self.strike_coverage / 0.6, 1.0)

        # Max gap penalty: 1 if gap < 5, 0 if gap > 50
        gap_score = max(0, 1 - (self.max_strike_gap - 5) / 45)

        # Balance: best if 50/50, worst if all one side
        if self.num_otm > 0:
            balance = min(self.num_calls, self.num_puts_synthetic) / (self.num_otm / 2)
            balance_score = min(balance, 1.0)
        else:
            balance_score = 0.0

        # Weighted average
        return 0.3 * n_score + 0.3 * cov_score + 0.2 * gap_score + 0.2 * balance_score


class ChainCleaner:
    """Clean and filter options chain data."""

    def __init__(
        self,
        min_volume: int = 10,
        min_oi: int = 100,
        max_spread_pct: float = 0.20,
        min_dte: int = 1,
        max_dte: int = 365,
    ):
        self.min_volume = min_volume
        self.min_oi = min_oi
        self.max_spread_pct = max_spread_pct
        self.min_dte = min_dte
        self.max_dte = max_dte

    def clean(
        self,
        chain: pd.DataFrame,
        trade_date: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Clean options chain data."""
        if chain.empty:
            return chain

        df = chain.copy()

        # Compute mid price
        if "mid" not in df.columns:
            df["mid"] = (df["bid"] + df["ask"]) / 2

        # Filter: Invalid quotes
        valid = (df["bid"] <= df["ask"]) & (df["bid"] >= 0) & (df["ask"] > 0)
        df = df[valid]

        # Filter: Minimum volume
        if self.min_volume > 0:
            df = df[df["volume"] >= self.min_volume]

        # Filter: Minimum open interest
        if self.min_oi > 0:
            df = df[df["open_interest"] >= self.min_oi]

        # Filter: Maximum spread
        if self.max_spread_pct > 0:
            spread = df["ask"] - df["bid"]
            mid = df["mid"].replace(0, np.nan)
            df = df[(spread / mid) <= self.max_spread_pct]

        # Filter: DTE range
        if trade_date is not None and "expiry" in df.columns:
            df["dte"] = (df["expiry"] - trade_date).dt.days
            df = df[(df["dte"] >= self.min_dte) & (df["dte"] <= self.max_dte)]

        return df.reset_index(drop=True)

    def filter_calls(self, chain: pd.DataFrame) -> pd.DataFrame:
        return chain[chain["right"] == "C"].copy()

    def filter_puts(self, chain: pd.DataFrame) -> pd.DataFrame:
        return chain[chain["right"] == "P"].copy()

    def build_otm_chain(
        self,
        chain: pd.DataFrame,
        spot: float,
        r: float = 0.05,
        return_quality: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, ChainQualityMetrics]]:
        """
        Build chain using OTM options only, converting puts to synthetic calls.

        Uses put-call parity: C = P + S - K*exp(-rT)

        For RND extraction, we want:
        - OTM calls (K > spot) - use directly
        - OTM puts (K < spot) - convert to synthetic call prices

        This gives better data quality since OTM options are more liquid.

        Args:
            chain: Cleaned options chain
            spot: Current spot price
            r: Risk-free rate
            return_quality: If True, also return quality metrics per expiry

        Returns:
            DataFrame with OTM options, or tuple of (DataFrame, quality_metrics_dict)
        """
        if chain.empty:
            if return_quality:
                return chain, {}
            return chain

        df = chain.copy()
        if "mid" not in df.columns:
            df["mid"] = (df["bid"] + df["ask"]) / 2

        # Need DTE for put-call parity
        if "dte" not in df.columns:
            raise ValueError("Chain must have 'dte' column. Run clean() first.")

        results = []
        quality_metrics = {}

        for expiry in df["expiry"].unique():
            exp_df = df[df["expiry"] == expiry]
            dte = exp_df["dte"].iloc[0]
            T = dte / 365.0

            num_raw = len(exp_df)
            num_calls = 0
            num_puts_synthetic = 0
            expiry_results = []

            # OTM calls (K > spot)
            otm_calls = exp_df[(exp_df["right"] == "C") & (exp_df["strike"] > spot)]
            for _, row in otm_calls.iterrows():
                expiry_results.append({
                    "expiry": expiry,
                    "strike": row["strike"],
                    "mid": row["mid"],
                    "source": "call",
                })
                num_calls += 1

            # OTM puts (K < spot) -> convert to synthetic call
            otm_puts = exp_df[(exp_df["right"] == "P") & (exp_df["strike"] < spot)]
            for _, row in otm_puts.iterrows():
                K = row["strike"]
                P = row["mid"]
                # Put-call parity: C = P + S - K*exp(-rT)
                discount = np.exp(-r * T)
                C = P + spot - K * discount
                if C > 0:
                    expiry_results.append({
                        "expiry": expiry,
                        "strike": K,
                        "mid": C,
                        "source": "put_synthetic",
                    })
                    num_puts_synthetic += 1

            # ATM: use call if available
            atm_calls = exp_df[
                (exp_df["right"] == "C") &
                (exp_df["strike"] >= spot * 0.99) &
                (exp_df["strike"] <= spot * 1.01)
            ]
            for _, row in atm_calls.iterrows():
                # Check if we already have this strike
                existing = [r for r in expiry_results if r["strike"] == row["strike"]]
                if not existing:
                    expiry_results.append({
                        "expiry": expiry,
                        "strike": row["strike"],
                        "mid": row["mid"],
                        "source": "call",
                    })
                    num_calls += 1

            results.extend(expiry_results)

            # Compute quality metrics for this expiry
            if expiry_results:
                strikes = np.array([r["strike"] for r in expiry_results])
                expiry_key = expiry.date() if hasattr(expiry, "date") else expiry
                quality_metrics[expiry_key] = ChainQualityMetrics(
                    num_raw=num_raw,
                    num_after_clean=len(exp_df),
                    num_otm=len(expiry_results),
                    num_calls=num_calls,
                    num_puts_synthetic=num_puts_synthetic,
                    strikes=strikes,
                    spot=spot,
                )

        if not results:
            if return_quality:
                return pd.DataFrame(columns=["expiry", "strike", "mid", "source"]), {}
            return pd.DataFrame(columns=["expiry", "strike", "mid", "source"])

        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values(["expiry", "strike"]).reset_index(drop=True)

        if return_quality:
            return result_df, quality_metrics
        return result_df
