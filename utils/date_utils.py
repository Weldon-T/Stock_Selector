from datetime import datetime, timedelta


def parse_date(date_str: str | None = None) -> str:
    """Parse input date to YYYYMMDD format. Returns today if None."""
    if date_str is None:
        return datetime.now().strftime("%Y%m%d")
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        try:
            dt = datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            raise ValueError(f"Invalid date format: {date_str!r}. Use YYYY-MM-DD or YYYYMMDD.")
    return dt.strftime("%Y%m%d")


def format_date(date_str: str) -> str:
    """Convert YYYYMMDD to YYYY-MM-DD for display."""
    dt = datetime.strptime(date_str, "%Y%m%d")
    return dt.strftime("%Y-%m-%d")


def get_latest_trade_date(trade_cal: list[str]) -> str | None:
    """Return the latest trade date <= today from a sorted list."""
    today = datetime.now().strftime("%Y%m%d")
    if not trade_cal:
        return None
    candidates = [d for d in trade_cal if d <= today]
    return candidates[-1] if candidates else None


def is_trade_date(date: str, trade_cal: list[str]) -> bool:
    """Check if date is in the trade calendar list."""
    return date in trade_cal


def get_date_range(start: str, end: str) -> list[str]:
    """Generate a list of dates between start and end (inclusive)."""
    start_dt = datetime.strptime(start, "%Y%m%d")
    end_dt = datetime.strptime(end, "%Y%m%d")
    dates = []
    current = start_dt
    while current <= end_dt:
        dates.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    return dates
