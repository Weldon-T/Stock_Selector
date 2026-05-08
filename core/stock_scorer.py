import numpy as np
import pandas as pd

from utils.logger import get_logger


class StockScorer:
    def __init__(self, config: dict):
        self.config = config
        self.select_count = config.get("select_count", 50)
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

    def compute_rank(self, series: pd.Series) -> pd.Series:
        """Percentile rank (0~1), higher is better raw value."""
        ranked = series.rank(pct=True, na_option="bottom")
        ranked = ranked.fillna(0.5)
        return ranked

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"Scoring {len(df)} stocks...")

        df = df.copy()

        # Compute percentile rank for each factor, then directionalize
        for f in self.factors:
            col = f["name"]
            rank_col = f"{col}_rank"
            if col in df.columns:
                df[rank_col] = self.compute_rank(df[col])
                if f["direction"] == "negative":
                    df[rank_col] = 1.0 - df[rank_col]
            else:
                df[rank_col] = np.nan

        # Weighted sum of rank columns
        df["final_score"] = 0.0
        for f in self.factors:
            rank_col = f"{f['name']}_rank"
            if rank_col in df.columns:
                df["final_score"] += df[rank_col].fillna(0.5) * f["weight"]

        df = df.sort_values("final_score", ascending=False)
        df = df.head(self.select_count)

        self.logger.info(f"Scoring complete: top {len(df)} stocks selected")
        return df.reset_index(drop=True)
