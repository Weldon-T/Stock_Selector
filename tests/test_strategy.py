import pytest
from main import _resolve_strategy


class TestResolveStrategy:
    def test_value_strategy_promotes_keys(self):
        config = {
            "strategies": {
                "value": {
                    "hold_months": 3,
                    "select_count": {"主板": 50, "创业板": 20, "科创板": 20},
                    "stock_pool": {"markets": ["主板"], "min_daily_amount": 50000},
                    "factors": {"value": {"ep_ttm": {"weight": 0.15, "enabled": True, "direction": "positive"}}},
                }
            }
        }
        _resolve_strategy(config, "value")

        assert config["factors"]["value"]["ep_ttm"]["weight"] == 0.15
        assert config["stock_pool"]["min_daily_amount"] == 50000
        assert config["select_count"]["主板"] == 50
        assert config["backtest"]["hold_months"] == 3

    def test_smallcap_different_hold_months(self):
        config = {
            "strategies": {
                "smallcap": {
                    "hold_months": 1,
                    "select_count": {"创业板": 30},
                    "stock_pool": {},
                    "factors": {},
                }
            }
        }
        _resolve_strategy(config, "smallcap")
        assert config["backtest"]["hold_months"] == 1
        assert config["select_count"]["创业板"] == 30

    def test_default_strategy_from_config(self):
        config = {
            "strategy": "value",
            "strategies": {
                "value": {"hold_months": 3, "factors": {"v": {}}},
                "smallcap": {"hold_months": 1, "factors": {"s": {}}},
            }
        }
        _resolve_strategy(config, None)
        assert "v" in config["factors"]

    def test_unknown_strategy_exits(self):
        config = {"strategies": {"value": {"factors": {}}}}
        with pytest.raises(SystemExit):
            _resolve_strategy(config, "nonexistent")

    def test_missing_strategies_noop(self):
        config = {"factors": {"old_format": {}}}
        _resolve_strategy(config, None)
        assert "old_format" in config["factors"]

    def test_original_keys_preserved(self):
        """Top-level shared config (paths, logging) untouched."""
        config = {
            "paths": {"output_dir": "./out"},
            "logging": {"level": "DEBUG"},
            "strategies": {
                "value": {"hold_months": 3, "factors": {"v": {}}},
            }
        }
        _resolve_strategy(config, "value")
        assert config["paths"]["output_dir"] == "./out"
        assert config["logging"]["level"] == "DEBUG"
