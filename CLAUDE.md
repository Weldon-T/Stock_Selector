# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git / GitHub

- 仓库地址：`https://github.com/Weldon-T/Stock_Selector`

## 项目概述

基于 A 股市场的轻量级全自动选股工具，融合基本面、技术面与动量因子，通过多因子打分模型筛选优质标的。
支持**双策略切换**——价值投资（质量+估值）和小盘优选（规模+反转+低波）。
使用 Tushare 免费 API（bak_basic + daily），结果输出为 CSV 文件。

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
│   ├── data_loader.py       # 缓存优先的数据拉取 (stock_basic/daily/multi_daily)
│   ├── filter.py            # ST/新股/停牌/板块/流动性过滤
│   ├── factor_calculator.py # 13因子计算 (基本面 + 技术面 + 动量)
│   ├── stock_scorer.py      # 百分位排名、分行业中性化、加权打分
│   └── backtest.py          # 季度调仓回测 + 多季度聚合
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
.venv/Scripts/python main.py                                      # 价值策略当日选股
.venv/Scripts/python main.py --strategy smallcap                  # 小盘策略当日选股
.venv/Scripts/python main.py --strategy value --date 2024-09-30   # 指定策略+历史日期
.venv/Scripts/python main.py --date 2026-01-20 --multi-quarter    # 多季度加权选股
.venv/Scripts/python main.py --backtest --start 2024-01-01 --end 2025-12-31 --strategy value
.venv/Scripts/python main.py --backtest --start 2024-01-01 --end 2025-12-31 --strategy smallcap
```

## 策略体系

所有策略共享 FactorCalculator（计算全部13个因子），仅权重/启用/持有期不同。配置在 `config.yaml` → `strategies`。

### value — 价值投资 (12因子, hold=3月, 配额 50/20/20)
重心：价值(25%) + 质量(35%) + 成长(16%)，short_reversal 禁用，流动性门槛 50M

### smallcap — 小盘优选 (13因子, hold=1月, 配额 30/30/30)
重心：规模(18%) + 反转(22%) + 低波/振幅/稳定(37%)，基本面压缩到 23%，流动性门槛 30M

## 关键机制

- **策略切换**：`--strategy value|smallcap`（默认 value）
- **分市场选股**：配额由 `select_count` dict 控制，各策略独立
- **分行业中性化**：百分位排名在行业内分别计算
- **多季度聚合**：基本面因子跨4季加权平均(0.2/0.2/0.3/0.3)；技术面因子仅用最新一季
- **流动性过滤**：阈值由各策略 `stock_pool.min_daily_amount` 独立控制
- **公平基准**：回测基准使用过滤后等权股票池

## 开发阶段

M1-M5 全部完成。当前分支 `small-cap-selection`：双策略架构 + 小盘优选。

## 设计原则

- **因子降级**：disabled 因子自动跳过，其权重按比例分配给同类别其他因子
- **缓存优先**：所有 Tushare 数据先查 SQLite 缓存，减少 API 调用
- **错误隔离**：每个因子独立计算，一个失败不影响其他
- **CSV 编码**：`utf-8-sig` 确保 Windows Excel 直接打开中文列名
- **请求限速**：Tushare API ≥0.2s 间隔，失败重试 3 次 (1s 退避)
- **数据缺失**：日频数据缺失则剔除该股票；季报空窗期沿用最近数据并标注 `financial_period`
