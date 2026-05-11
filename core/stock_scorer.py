import numpy as np
import pandas as pd

from utils.logger import get_logger


class StockScorer:
    def __init__(self, config: dict):
        self.config = config
        self.factors: list[dict] = []
        self.logger = get_logger()
        self._build_factor_config()

    def _build_factor_config(self):
        factors_config = self.config.get("factors", {})
        for category, cat_cfg in factors_config.items():
            if not isinstance(cat_cfg, dict):
                continue
            for name, cfg in cat_cfg.items():
                if not isinstance(cfg, dict):
                    continue
                if cfg.get("enabled", True):
                    self.factors.append({
                        "name": name,
                        "weight": abs(cfg["weight"]),
                        "direction": cfg.get("direction", "positive"),
                        "category": category,
                    })

        total_weight = sum(f["weight"] for f in self.factors)
        if total_weight > 0:
            for f in self.factors:
                f["weight"] = f["weight"] / total_weight

        enabled_names = [f["name"] for f in self.factors]
        self.logger.info(f"Enabled factors ({len(self.factors)}): {enabled_names}")
        weight_info = [f"{f['name']}={f['weight']:.3f}" for f in self.factors]
        self.logger.info(f"Normalized weights: {weight_info}")

    def _grouped_rank(self, df: pd.DataFrame, col: str, group_col: str) -> pd.Series:
        """Percentile rank within each group. Returns Series aligned to df index."""
        result = pd.Series(np.nan, index=df.index)
        if group_col not in df.columns:
            # Fall back to cross-market ranking
            ranked = df[col].rank(pct=True, na_option="bottom")
            return ranked.fillna(0.5)
        for g, idx in df.groupby(group_col).groups.items():
            subset = df.loc[idx, col]
            ranked = subset.rank(pct=True, na_option="bottom")
            result.loc[idx] = ranked
        return result.fillna(0.5)

    def _compute_ranks(self, df: pd.DataFrame, sector_neutral: bool = False) -> pd.DataFrame:
        """Compute percentile ranks and final_score for all stocks."""
        df = df.copy()

        group_col = "market" if sector_neutral else None

        for f in self.factors:
            col = f["name"]
            rank_col = f"{col}_rank"
            if col in df.columns:
                if sector_neutral and group_col and group_col in df.columns:
                    df[rank_col] = self._grouped_rank(df, col, group_col)
                else:
                    df[rank_col] = df[col].rank(pct=True, na_option="bottom").fillna(0.5)

                if f["direction"] == "negative":
                    df[rank_col] = 1.0 - df[rank_col]
            else:
                df[rank_col] = np.nan

        df["final_score"] = 0.0
        for f in self.factors:
            rank_col = f"{f['name']}_rank"
            if rank_col in df.columns:
                df["final_score"] += df[rank_col].fillna(0.5) * f["weight"]

        return df.sort_values("final_score", ascending=False)

    def score_all(self, df: pd.DataFrame, sector_neutral: bool = False) -> pd.DataFrame:
        """Compute ranks for all stocks without top-N truncation."""
        return self._compute_ranks(df, sector_neutral).reset_index(drop=True)

