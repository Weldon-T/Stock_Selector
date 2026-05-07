import sqlite3
from pathlib import Path

import pandas as pd

from utils.logger import get_logger


class SQLiteCache:
    def __init__(self, cache_dir: str = "./cache"):
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self.db_path = Path(cache_dir) / "stock_cache.db"
        self.logger = get_logger()
        self._init_db()

    def _init_db(self):
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS cache_meta (
                        table_name TEXT NOT NULL,
                        date_key  TEXT NOT NULL,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (table_name, date_key)
                    )
                    """
                )
                conn.commit()
        except sqlite3.Error as e:
            self.logger.error(f"Failed to init cache DB: {e}")

    def _safe_table_name(self, table: str, date: str) -> str:
        name = f"data_{table}_{date}"
        return name.replace("-", "_").replace(".", "_")

    def has(self, table: str, date: str) -> bool:
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cur = conn.execute(
                    "SELECT 1 FROM cache_meta WHERE table_name = ? AND date_key = ?",
                    (table, date),
                )
                return cur.fetchone() is not None
        except sqlite3.Error:
            return False

    def get(self, table: str, date: str) -> pd.DataFrame | None:
        tbl = self._safe_table_name(table, date)
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                return pd.read_sql(f"SELECT * FROM [{tbl}]", conn)
        except (sqlite3.Error, pd.errors.DatabaseError):
            return None

    def put(self, table: str, date: str, data: pd.DataFrame) -> None:
        if data.empty:
            return
        tbl = self._safe_table_name(table, date)
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                data.to_sql(tbl, conn, if_exists="replace", index=False)
                conn.execute(
                    "INSERT OR REPLACE INTO cache_meta (table_name, date_key, updated_at) "
                    "VALUES (?, ?, CURRENT_TIMESTAMP)",
                    (table, date),
                )
                conn.commit()
            self.logger.info(f"Cached {table}/{date}: {len(data)} rows")
        except sqlite3.Error as e:
            self.logger.warning(f"Cache write failed for {table}/{date}: {e}")

    def get_latest_date(self, table: str) -> str | None:
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cur = conn.execute(
                    "SELECT date_key FROM cache_meta WHERE table_name = ? "
                    "ORDER BY date_key DESC LIMIT 1",
                    (table,),
                )
                row = cur.fetchone()
                return row[0] if row else None
        except sqlite3.Error:
            return None
