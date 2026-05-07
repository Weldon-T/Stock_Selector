import pandas as pd

from utils.logger import get_logger


class Backtest:
    def __init__(self, config: dict):
        bt_cfg = config.get("backtest", {})
        self.hold_period = bt_cfg.get("hold_period", 20)
        self.logger = get_logger()

    def run(self, start_date: str, end_date: str) -> pd.DataFrame:
        self.logger.info(f"Backtest mode: {start_date} ~ {end_date}, hold={self.hold_period}d")
        self.logger.info("Backtest module is a stub — full implementation in M5")
        return pd.DataFrame(columns=["date", "return", "sharpe", "max_drawdown", "win_rate"])
