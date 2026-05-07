# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git / GitHub

- 仓库地址：`https://github.com/Weldon-T/Stock_Selector`

## 项目概述

基于 A 股市场的轻量级全自动选股工具，融合基本面与资金面因子，通过多因子打分模型筛选优质标的。使用 Tushare 免费 API，结果输出为 CSV 文件。

## 技术栈

- Python 3.10+
- 虚拟环境：`.venv/`（项目根目录）
- 依赖：tushare, pandas, numpy, pyyaml
- Python 路径：`.venv/Scripts/python`

## 项目结构

```
Stock_Selector/
├── main.py                  # 主入口 (CLI: --date / --backtest)
├── config.yaml              # 配置文件 (因子权重、tushare_token 等)
├── requirements.txt         # Python 依赖
├── core/
│   ├── tushare_client.py    # Tushare 登录、限速(0.2s)、重试(3次)
│   ├── data_loader.py       # 缓存优先的数据拉取 (stock_basic/daily/trade_cal)
│   ├── filter.py            # ST/新股(<60天)/停牌(>10天)/板块过滤
│   ├── factor_calculator.py # 因子原始值计算 (M1 桩: PE/PB 合并)
│   ├── stock_scorer.py      # 百分位排名、加权打分 (M1 桩: 随机)
│   └── backtest.py          # 回测模块 (M1 桩)
├── utils/
│   ├── cache.py             # SQLite 缓存 (SQLiteCache 类)
│   ├── date_utils.py        # 日期解析/格式化工具
│   └── logger.py            # 日志 (文件 UTF-8 + 控制台双输出)
├── logs/                    # 运行日志 (run.log)
├── output/                  # CSV 输出 (选股结果_YYYYMMDD.csv, UTF-8 BOM)
└── cache/                   # SQLite 缓存文件 (stock_cache.db)
```

## 核心流水线

```
加载 config.yaml → 日志初始化 → TushareClient 登录 → DataLoader 拉取(缓存优先)
→ StockFilter 过滤 → FactorCalculator 计算因子 → StockScorer 打分排序
→ 输出 CSV (UTF-8 BOM) → 控制台打印 Top10
```

## CLI 命令

```bash
.venv/Scripts/python main.py                                      # 当日选股
.venv/Scripts/python main.py --date 2024-09-30                    # 历史选股
.venv/Scripts/python main.py --backtest --start 2024-01-01 --end 2024-12-31  # 回测
```

## 因子体系

### 基本面因子 (fundamental)
| 因子 | 权重 | 方向 | 数据来源 |
|------|------|------|----------|
| pe_ttm | -0.20 | negative | daily.pe_ttm |
| pb | -0.10 | negative | daily.pb |
| roe_ttm | 0.20 | positive | income/balancesheet |
| net_profit_yoy | 0.15 | positive | income |
| revenue_yoy | 0.10 | positive | income |
| debt_to_asset | -0.05 | negative | balancesheet |

### 资金面因子 (capital_flow)
| 因子 | 权重 | 方向 | 数据来源 | 状态 |
|------|------|------|----------|------|
| volume_ratio | 0.10 | positive | daily 计算 | 启用 |
| margin_chg_5d | 0.05 | positive | margin | 启用 |
| main_inflow_5d | 0.10 | positive | moneyflow | 需积分，默认关闭 |
| north_net_inflow | 0.05 | positive | hsgt/hk_hold | 需积分，默认关闭 |

## 开发阶段 (里程碑)

| 阶段 | 内容 | 状态 |
|------|------|------|
| M1 | 项目结构、Tushare接入、基础行情获取 | ✅ 完成 |
| M2 | 基本面因子计算 (PE/PB/ROE/成长/负债) | ⏳ 待实现 |
| M3 | 资金面因子计算 (量比、融资、可选资金流) | ⏳ 待实现 |
| M4 | 百分位打分、过滤、CSV输出、日志 | ⏳ 待实现 |
| M5 | 配置系统完善、回测模块、文档 | ⏳ 待实现 |

## 设计原则

- **因子降级**：disabled 因子自动跳过，其权重按比例分配给同类别其他因子
- **缓存优先**：所有 Tushare 数据先查 SQLite 缓存，减少 API 调用
- **错误隔离**：每个因子独立计算，一个失败不影响其他
- **CSV 编码**：`utf-8-sig` 确保 Windows Excel 直接打开中文列名
- **请求限速**：Tushare API ≥0.2s 间隔，失败重试 3 次 (1s 退避)
- **数据缺失**：日频数据缺失则剔除该股票；季报空窗期沿用最近数据并标注 `financial_period`
