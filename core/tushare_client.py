import time

import pandas as pd
import tushare as ts

from utils.logger import get_logger


class TushareClient:
    def __init__(self, token: str, min_interval: float = 0.2):
        self.token = token
        self.min_interval = min_interval
        self._api = None
        self._initialized = False
        self._last_request_time = 0.0
        self.logger = get_logger()

    def login(self) -> bool:
        try:
            ts.set_token(self.token)
            self._api = ts.pro_api()
            df = self._api.trade_cal(exchange="SSE", start_date="20240101", end_date="20240110")
            if df is not None and not df.empty:
                self._initialized = True
                self.logger.info("Tushare login successful")
                return True
        except Exception as e:
            self.logger.error(f"Tushare login failed: {e}")
        self.logger.error("Tushare login failed — check your token and network")
        return False

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request_time = time.time()

    def request(self, api_name: str, fields: str = "", max_retries: int = 3, **params) -> pd.DataFrame:
        if not self._initialized:
            self.logger.warning(f"Client not initialized, skipping {api_name}")
            return pd.DataFrame()

        for attempt in range(max_retries):
            self._rate_limit()
            try:
                func = getattr(self._api, api_name, None)
                if func is None:
                    self.logger.error(f"Unknown API: {api_name}")
                    return pd.DataFrame()

                if fields:
                    result = func(fields=fields, **params)
                else:
                    result = func(**params)

                if result is not None and not result.empty:
                    return result

                self.logger.warning(f"{api_name} returned empty (attempt {attempt + 1}/{max_retries})")

            except Exception as e:
                self.logger.warning(f"{api_name} error (attempt {attempt + 1}/{max_retries}): {e}")

            if attempt < max_retries - 1:
                time.sleep(1)

        return pd.DataFrame()

    def stock_basic(self) -> pd.DataFrame:
        fields = "ts_code,symbol,name,area,industry,list_date,market,exchange,is_st"
        return self.request("stock_basic", fields=fields)

    def daily(self, trade_date: str) -> pd.DataFrame:
        return self.request("daily", trade_date=trade_date)

    def trade_cal(self, start_date: str, end_date: str) -> pd.DataFrame:
        return self.request("trade_cal", exchange="SSE", start_date=start_date, end_date=end_date)

    def income(self, period: str) -> pd.DataFrame:
        fields = (
            "ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,"
            "revenue,oper_profit,total_profit,income_tax,n_income,"
            "n_income_attr_p,basic_eps,diluted_eps"
        )
        return self.request("income", fields=fields, period=period)

    def balancesheet(self, period: str) -> pd.DataFrame:
        fields = (
            "ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,"
            "total_assets,total_liab,total_hldr_eqy_exc_min_int"
        )
        return self.request("balancesheet", fields=fields, period=period)

    def margin(self, trade_date: str) -> pd.DataFrame:
        return self.request("margin", trade_date=trade_date)

    def moneyflow(self, trade_date: str) -> pd.DataFrame:
        return self.request("moneyflow", trade_date=trade_date)

    def fina_indicator(self, period: str) -> pd.DataFrame:
        return self.request("fina_indicator", period=period)
