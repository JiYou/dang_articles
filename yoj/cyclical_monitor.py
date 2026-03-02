#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
周期股监控系统 — 基于 MR Dang 投资哲学

功能概述：
    1. 商品期货周期仪表盘（铜、铝、金、银、原油、螺纹钢等）
    2. 股票池实时行情与持仓监控
    3. 金字塔建仓信号（补仓 / 止盈 / 止损 / 板块集中度）
    4. 标准 OHLCV 数据接口，可供外部技术指标系统调用

使用方法：
    python cyclical_monitor.py               # 默认生成每日报告
    python cyclical_monitor.py --daily       # 每日报告
    python cyclical_monitor.py --weekly      # 每周报告
    python cyclical_monitor.py --futures     # 商品期货仪表盘
    python cyclical_monitor.py --stock 601899  # 单只股票详情
    python cyclical_monitor.py --data 601899 2025-01-01 2025-12-31  # 导出数据

依赖：
    pip install akshare pandas

作者：基于《周期股投资方案》与《交易三大纪律》构建
"""

import argparse
import datetime
import sys
import time
import traceback
from typing import Optional

import akshare as ak
import pandas as pd

# ========== 配置区域 ==========

# 周期股票池（可自行修改）
# entry_price: 买入均价（None 表示未持仓）
# shares: 持仓股数（0 表示未持仓）
STOCK_POOL = {
    # 铜板块
    "601899": {"name": "紫金矿业", "sector": "铜", "entry_price": None, "shares": 0},
    "601168": {"name": "西部矿业", "sector": "铜", "entry_price": None, "shares": 0},
    "600362": {"name": "江西铜业", "sector": "铜", "entry_price": None, "shares": 0},
    # 铝板块
    "600219": {"name": "南山铝业", "sector": "铝", "entry_price": None, "shares": 0},
    "000933": {"name": "神火股份", "sector": "铝", "entry_price": None, "shares": 0},
    # 煤化工
    "600989": {"name": "宝丰能源", "sector": "煤化工", "entry_price": None, "shares": 0},
    # 钾肥
    "000792": {"name": "盐湖股份", "sector": "钾肥/锂", "entry_price": None, "shares": 0},
    # 航运
    "601919": {"name": "中远海控", "sector": "航运", "entry_price": None, "shares": 0},
    # 石油
    "600938": {"name": "中国海油", "sector": "石油", "entry_price": None, "shares": 0},
}

# 关键期货品种（用于监控商品周期）
FUTURES_CODES = {
    "CU0": "沪铜主力",
    "AL0": "沪铝主力",
    "AU0": "沪金主力",
    "AG0": "沪银主力",
    "SC0": "原油主力",
    "RB0": "螺纹钢主力",
    "JM0": "焦煤主力",
    "PP0": "聚丙烯主力",
    "MA0": "甲醇主力",
    "LH0": "生猪主力",
}

# MR Dang 仓位管理参数
MAX_SINGLE_STOCK_PCT = 0.20   # 个股最高仓位 20%
INITIAL_POSITION_PCT = 0.15   # 初始建仓比例 15%
ADD_TRIGGER_DROP_PCT = 0.10   # 补仓触发跌幅 10%
TAKE_PROFIT_PCT = 0.30        # 止盈线 30%
STOP_LOSS_PCT = 0.20          # 硬止损 20%
MAX_SECTOR_PCT = 0.40         # 单板块最高仓位 40%

# 打印分隔线宽度
LINE_WIDTH = 72


# ========== 工具函数 ==========

def _print_header(title: str) -> None:
    """打印带分隔线的标题"""
    print()
    print("=" * LINE_WIDTH)
    print(f"  {title}")
    print("=" * LINE_WIDTH)


def _print_section(title: str) -> None:
    """打印小节标题"""
    print()
    print(f"--- {title} ---")


def _safe_call(func, *args, **kwargs):
    """安全调用 akshare 接口，捕获异常并打印错误信息"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"[错误] 调用 {func.__name__} 失败: {e}")
        traceback.print_exc()
        return None


def _pct_str(value: float) -> str:
    """将小数转换为百分比字符串，带颜色提示符号"""
    if value > 0:
        return f"+{value:.2f}%"
    else:
        return f"{value:.2f}%"


def _trend_label(pct: float) -> str:
    """根据涨跌幅判断简单趋势"""
    if pct > 5:
        return "↑ 上涨"
    elif pct < -5:
        return "↓ 下跌"
    else:
        return "→ 盘整"


def _market_prefix(stock_code: str) -> str:
    """根据股票代码判断市场前缀（sh / sz）"""
    if stock_code.startswith("6"):
        return "sh"
    elif stock_code.startswith("0") or stock_code.startswith("3"):
        return "sz"
    else:
        return "sh"


# ========== 核心数据获取函数 ==========

def get_stock_daily(stock_code: str, days: int = 120) -> Optional[pd.DataFrame]:
    """获取个股日线数据（前复权）

    Args:
        stock_code: 股票代码，如 "601899"
        days: 获取最近多少个交易日的数据

    Returns:
        DataFrame，列: date, open, close, high, low, volume, turnover
        如果获取失败返回 None
    """
    # 计算起始日期（多取一些天数以覆盖非交易日）
    end_date = datetime.date.today().strftime("%Y%m%d")
    start_date = (datetime.date.today() - datetime.timedelta(days=days * 2)).strftime("%Y%m%d")

    df = _safe_call(
        ak.stock_zh_a_hist,
        symbol=stock_code,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust="qfq",
    )
    if df is None or df.empty:
        return None

    # 统一列名
    df = df.rename(columns={
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "turnover",
    })

    # 只保留需要的列（akshare 返回的列可能包含其他字段）
    keep_cols = ["date", "open", "close", "high", "low", "volume", "turnover"]
    available_cols = [c for c in keep_cols if c in df.columns]
    df = df[available_cols]

    # 取最后 N 条
    df = df.tail(days).reset_index(drop=True)
    return df


def get_stock_realtime(stock_codes: list) -> Optional[pd.DataFrame]:
    """获取实时行情快照

    Args:
        stock_codes: 股票代码列表，如 ["601899", "601168"]

    Returns:
        DataFrame，包含代码、名称、最新价、涨跌幅、成交量等
        如果获取失败返回 None
    """
    df = _safe_call(ak.stock_zh_a_spot_em)
    if df is None or df.empty:
        return None

    # 筛选目标股票
    df = df[df["代码"].isin(stock_codes)].copy()
    if df.empty:
        print("[警告] 未能从实时行情中筛选到目标股票")
        return None

    # 选取关键列
    cols_map = {
        "代码": "code",
        "名称": "name",
        "最新价": "price",
        "涨跌幅": "change_pct",
        "涨跌额": "change_amt",
        "成交量": "volume",
        "成交额": "turnover",
        "今开": "open",
        "最高": "high",
        "最低": "low",
        "昨收": "prev_close",
        "换手率": "turnover_rate",
    }
    available_rename = {k: v for k, v in cols_map.items() if k in df.columns}
    df = df.rename(columns=available_rename)
    available_cols = [v for v in available_rename.values()]
    df = df[available_cols].reset_index(drop=True)
    return df


def get_futures_data(futures_code: str, days: int = 60) -> Optional[pd.DataFrame]:
    """获取期货日线数据

    Args:
        futures_code: 期货代码，如 "CU0"
        days: 获取最近多少个交易日的数据

    Returns:
        DataFrame，列: date, open, high, low, close, volume
        如果获取失败返回 None
    """
    df = _safe_call(ak.futures_zh_daily_sina, symbol=futures_code)
    if df is None or df.empty:
        return None

    # 统一列名
    rename_map = {
        "date": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }
    # 部分版本可能列名不同，做容错处理
    for old_name, new_name in rename_map.items():
        if old_name in df.columns and old_name != new_name:
            df = df.rename(columns={old_name: new_name})

    df = df.tail(days).reset_index(drop=True)
    return df


def get_stock_financials(stock_code: str) -> Optional[dict]:
    """获取关键财务指标（PE, PB, ROE 等）

    Args:
        stock_code: 股票代码

    Returns:
        字典，包含 pe, pb, roe 等关键指标；获取失败返回 None
    """
    df = _safe_call(ak.stock_financial_analysis_indicator, symbol=stock_code)
    if df is None or df.empty:
        return None

    # 取最新一期数据
    latest = df.iloc[0]

    result = {}
    # 不同版本的 akshare 列名可能有差异，做容错
    field_map = {
        "摊薄每股收益(元)": "eps",
        "加权净资产收益率(%)": "roe",
        "每股净资产_调整后(元)": "bvps",
        "流动比率": "current_ratio",
        "速动比率": "quick_ratio",
    }
    for cn_name, en_name in field_map.items():
        if cn_name in latest.index:
            try:
                result[en_name] = float(latest[cn_name])
            except (ValueError, TypeError):
                result[en_name] = None

    return result


def get_dividend_history(stock_code: str) -> Optional[pd.DataFrame]:
    """获取分红历史

    Args:
        stock_code: 股票代码

    Returns:
        DataFrame，包含历年分红记录；获取失败返回 None
    """
    df = _safe_call(
        ak.stock_history_dividend_detail,
        symbol=stock_code,
        indicator="分红",
    )
    if df is None or df.empty:
        return None
    return df


def get_fund_flow(stock_code: str) -> Optional[pd.DataFrame]:
    """获取资金流向（近期）

    Args:
        stock_code: 股票代码

    Returns:
        DataFrame，包含主力/超大单/大单/中单/小单资金流向数据
        获取失败返回 None
    """
    market = _market_prefix(stock_code)
    df = _safe_call(
        ak.stock_individual_fund_flow,
        stock=stock_code,
        market=market,
    )
    if df is None or df.empty:
        return None
    return df


# ========== 信号检测函数 ==========

def _get_portfolio_total_value(portfolio: dict, realtime_prices: dict) -> float:
    """计算持仓总市值（用于仓位比例计算）

    Args:
        portfolio: STOCK_POOL 格式的持仓字典
        realtime_prices: {股票代码: 当前价格} 的字典

    Returns:
        总持仓市值（浮点数）
    """
    total = 0.0
    for code, info in portfolio.items():
        if info["entry_price"] is not None and info["shares"] > 0:
            price = realtime_prices.get(code, info["entry_price"])
            total += price * info["shares"]
    return total


def check_add_position_signals(portfolio: dict, realtime_prices: dict) -> list:
    """检查补仓信号

    对持仓股票，检查是否触发补仓条件：
    - 从买入价下跌超过 10%
    - 且当前仓位未超过 20% 上限

    Args:
        portfolio: STOCK_POOL 格式的持仓字典
        realtime_prices: {股票代码: 当前价格}

    Returns:
        信号列表，每个元素为字典，包含 stock, name, drop_pct, action, reason
    """
    signals = []
    total_value = _get_portfolio_total_value(portfolio, realtime_prices)

    for code, info in portfolio.items():
        if info["entry_price"] is None or info["shares"] <= 0:
            continue

        current_price = realtime_prices.get(code)
        if current_price is None:
            continue

        # 计算跌幅
        drop_pct = (current_price - info["entry_price"]) / info["entry_price"]

        if drop_pct <= -ADD_TRIGGER_DROP_PCT:
            # 计算当前仓位占比
            position_value = current_price * info["shares"]
            position_pct = position_value / total_value if total_value > 0 else 0

            if position_pct < MAX_SINGLE_STOCK_PCT:
                signals.append({
                    "stock": code,
                    "name": info["name"],
                    "sector": info["sector"],
                    "drop_pct": drop_pct * 100,
                    "current_position_pct": position_pct * 100,
                    "action": "可补仓",
                    "reason": (
                        f"跌幅 {drop_pct * 100:.1f}% 超过 {ADD_TRIGGER_DROP_PCT * 100:.0f}% 触发线，"
                        f"当前仓位 {position_pct * 100:.1f}% 未超限"
                    ),
                })
            else:
                signals.append({
                    "stock": code,
                    "name": info["name"],
                    "sector": info["sector"],
                    "drop_pct": drop_pct * 100,
                    "current_position_pct": position_pct * 100,
                    "action": "仓位已满",
                    "reason": (
                        f"跌幅 {drop_pct * 100:.1f}% 已触发补仓，"
                        f"但仓位 {position_pct * 100:.1f}% 已达上限 {MAX_SINGLE_STOCK_PCT * 100:.0f}%"
                    ),
                })

    return signals


def check_take_profit_signals(portfolio: dict, realtime_prices: dict) -> list:
    """检查止盈信号

    对持仓股票，检查是否触发止盈条件：
    - 从买入价上涨超过 30%

    Args:
        portfolio: STOCK_POOL 格式的持仓字典
        realtime_prices: {股票代码: 当前价格}

    Returns:
        信号列表
    """
    signals = []

    for code, info in portfolio.items():
        if info["entry_price"] is None or info["shares"] <= 0:
            continue

        current_price = realtime_prices.get(code)
        if current_price is None:
            continue

        gain_pct = (current_price - info["entry_price"]) / info["entry_price"]

        if gain_pct >= TAKE_PROFIT_PCT:
            signals.append({
                "stock": code,
                "name": info["name"],
                "sector": info["sector"],
                "gain_pct": gain_pct * 100,
                "action": "建议止盈",
                "reason": (
                    f"浮盈 {gain_pct * 100:.1f}% 已达止盈线 {TAKE_PROFIT_PCT * 100:.0f}%，"
                    f"积小胜为大胜，分批撤退"
                ),
            })

    return signals


def check_stop_loss_signals(portfolio: dict, realtime_prices: dict) -> list:
    """检查止损信号

    对持仓股票，检查是否触发硬止损：
    - 从买入价下跌超过 20%

    Args:
        portfolio: STOCK_POOL 格式的持仓字典
        realtime_prices: {股票代码: 当前价格}

    Returns:
        信号列表
    """
    signals = []

    for code, info in portfolio.items():
        if info["entry_price"] is None or info["shares"] <= 0:
            continue

        current_price = realtime_prices.get(code)
        if current_price is None:
            continue

        drop_pct = (current_price - info["entry_price"]) / info["entry_price"]

        if drop_pct <= -STOP_LOSS_PCT:
            signals.append({
                "stock": code,
                "name": info["name"],
                "sector": info["sector"],
                "drop_pct": drop_pct * 100,
                "action": "⚠️ 触发硬止损",
                "reason": (
                    f"浮亏 {drop_pct * 100:.1f}% 已超过硬止损线 {STOP_LOSS_PCT * 100:.0f}%，"
                    f"必须评估：是市场错杀还是看走眼了？"
                ),
            })

    return signals


def check_sector_concentration(portfolio: dict, realtime_prices: dict) -> list:
    """检查板块集中度

    确保单一板块仓位不超过 40%

    Args:
        portfolio: STOCK_POOL 格式的持仓字典
        realtime_prices: {股票代码: 当前价格}

    Returns:
        警告列表
    """
    warnings = []
    total_value = _get_portfolio_total_value(portfolio, realtime_prices)

    if total_value <= 0:
        return warnings

    # 按板块汇总
    sector_values = {}
    for code, info in portfolio.items():
        if info["entry_price"] is None or info["shares"] <= 0:
            continue
        price = realtime_prices.get(code, info["entry_price"])
        value = price * info["shares"]
        sector = info["sector"]
        sector_values[sector] = sector_values.get(sector, 0) + value

    for sector, value in sector_values.items():
        pct = value / total_value
        if pct > MAX_SECTOR_PCT:
            warnings.append({
                "sector": sector,
                "pct": pct * 100,
                "action": "⚠️ 板块超配",
                "reason": (
                    f"{sector}板块仓位 {pct * 100:.1f}% 超过上限 {MAX_SECTOR_PCT * 100:.0f}%，"
                    f"需要减仓或分散到其他板块"
                ),
            })

    return warnings


# ========== 商品周期仪表盘 ==========

def commodity_cycle_dashboard() -> None:
    """商品周期仪表盘

    打印关键商品期货的近期走势概要：
    - 当前价格
    - 近一周 / 一月 / 三月涨跌幅
    - 简单趋势判断（上涨 / 盘整 / 下跌）
    """
    _print_header("商品期货周期仪表盘")

    print(f"\n{'品种':<12} {'最新价':>10} {'周涨跌':>10} {'月涨跌':>10} {'季涨跌':>10} {'趋势':>8}")
    print("-" * LINE_WIDTH)

    for code, name in FUTURES_CODES.items():
        df = get_futures_data(code, days=90)
        if df is None or len(df) < 5:
            print(f"{name:<12} {'数据获取失败':>10}")
            continue

        # 确保 close 列为数值类型
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])

        if df.empty:
            print(f"{name:<12} {'数据为空':>10}")
            continue

        latest_price = df["close"].iloc[-1]

        # 一周涨跌（约 5 个交易日）
        week_idx = max(0, len(df) - 5)
        week_pct = (latest_price - df["close"].iloc[week_idx]) / df["close"].iloc[week_idx] * 100

        # 一月涨跌（约 22 个交易日）
        month_idx = max(0, len(df) - 22)
        month_pct = (latest_price - df["close"].iloc[month_idx]) / df["close"].iloc[month_idx] * 100

        # 三月涨跌（全部数据）
        quarter_pct = (latest_price - df["close"].iloc[0]) / df["close"].iloc[0] * 100

        # 趋势判断（基于月涨跌）
        trend = _trend_label(month_pct)

        print(
            f"{name:<12} {latest_price:>10.2f} "
            f"{_pct_str(week_pct):>10} {_pct_str(month_pct):>10} "
            f"{_pct_str(quarter_pct):>10} {trend:>8}"
        )

        # 请求间隔，避免被限流
        time.sleep(0.3)

    print()
    print("💡 提示：期货走势是股票的先行指标。铜铝金上涨通常利好有色板块。")


# ========== 报告函数 ==========

def _print_realtime_positions() -> None:
    """打印持仓股票的实时状态表"""
    _print_section("持仓股票实时状态")

    stock_codes = list(STOCK_POOL.keys())
    df = get_stock_realtime(stock_codes)

    if df is None:
        print("[警告] 无法获取实时行情，可能是非交易时间")
        return

    # 打印表头
    print(f"\n{'代码':<8} {'名称':<10} {'板块':<8} {'最新价':>8} {'涨跌幅':>8} "
          f"{'买入价':>8} {'浮盈亏':>10} {'持股':>8}")
    print("-" * LINE_WIDTH)

    for _, row in df.iterrows():
        code = row.get("code", "")
        info = STOCK_POOL.get(code, {})
        name = info.get("name", row.get("name", ""))
        sector = info.get("sector", "")
        price = row.get("price", 0)
        change_pct = row.get("change_pct", 0)

        entry_price = info.get("entry_price")
        shares = info.get("shares", 0)

        # 浮盈亏
        if entry_price is not None and shares > 0 and price > 0:
            pnl = (price - entry_price) * shares
            pnl_str = f"{pnl:>+.0f}"
            entry_str = f"{entry_price:.2f}"
            shares_str = f"{shares}"
        else:
            pnl_str = "-"
            entry_str = "-"
            shares_str = "-"

        try:
            change_str = _pct_str(float(change_pct))
        except (ValueError, TypeError):
            change_str = "-"

        try:
            price_str = f"{float(price):.2f}"
        except (ValueError, TypeError):
            price_str = "-"

        print(
            f"{code:<8} {name:<10} {sector:<8} {price_str:>8} {change_str:>8} "
            f"{entry_str:>8} {pnl_str:>10} {shares_str:>8}"
        )


def _get_realtime_prices() -> dict:
    """获取实时价格字典，用于信号检测

    Returns:
        {股票代码: 当前价格} 的字典
    """
    stock_codes = list(STOCK_POOL.keys())
    df = get_stock_realtime(stock_codes)
    prices = {}

    if df is not None:
        for _, row in df.iterrows():
            code = row.get("code", "")
            try:
                prices[code] = float(row.get("price", 0))
            except (ValueError, TypeError):
                pass

    return prices


def _print_signals(signals: list, title: str) -> None:
    """打印信号列表"""
    _print_section(title)
    if not signals:
        print("  无信号")
        return
    for sig in signals:
        stock_label = f"{sig.get('name', '')}({sig.get('stock', '')})"
        action = sig.get("action", "")
        reason = sig.get("reason", "")
        print(f"  [{action}] {stock_label}")
        print(f"         {reason}")


def _has_positions() -> bool:
    """检查是否有任何持仓"""
    return any(
        info["entry_price"] is not None and info["shares"] > 0
        for info in STOCK_POOL.values()
    )


def daily_report() -> None:
    """每日报告

    综合输出：
    1. 商品期货概况
    2. 持仓股票实时状态
    3. 补仓 / 止盈 / 止损信号
    4. 板块集中度检查
    5. 今日操作建议
    """
    today = datetime.date.today().strftime("%Y-%m-%d")
    _print_header(f"周期股每日监控报告 — {today}")

    # 1. 商品期货概况
    commodity_cycle_dashboard()

    # 2. 持仓股票实时状态
    _print_realtime_positions()

    # 3-4. 信号检测（仅在有持仓时执行）
    if _has_positions():
        prices = _get_realtime_prices()
        if prices:
            # 补仓信号
            add_signals = check_add_position_signals(STOCK_POOL, prices)
            _print_signals(add_signals, "补仓信号检测")

            # 止盈信号
            tp_signals = check_take_profit_signals(STOCK_POOL, prices)
            _print_signals(tp_signals, "止盈信号检测")

            # 止损信号
            sl_signals = check_stop_loss_signals(STOCK_POOL, prices)
            _print_signals(sl_signals, "止损信号检测")

            # 板块集中度
            sector_warnings = check_sector_concentration(STOCK_POOL, prices)
            _print_signals(sector_warnings, "板块集中度检查")
        else:
            print("\n[警告] 无法获取实时价格，跳过信号检测")
    else:
        _print_section("信号检测")
        print("  当前无持仓，信号检测已跳过")
        print("  💡 请在 STOCK_POOL 中设置 entry_price 和 shares 以启用信号监控")

    # 5. 操作建议
    _print_section("今日操作建议")
    print("  1. 检查外盘大宗商品（美铜、伦铝、黄金、原油）的隔夜走势")
    print("  2. 只关注预设的报警位，不要盯盘")
    print("  3. 如有触发信号，先确认基本面是否变化再行动")
    print("  4. 任何操作必须记录在交易日志中")
    print()
    print("  ⚠️ 三大纪律提醒：")
    print("     - 单笔亏损不超过总资本的 2%")
    print("     - 先有计划，后有动作，杜绝临时起意")
    print("     - 一致性执行，不因单次盈亏改变策略")

    print()
    print(f"报告生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * LINE_WIDTH)


def weekly_report() -> None:
    """每周报告

    在 daily_report 基础上增加：
    1. 本周持仓表更新（成本、仓位占比、浮盈浮亏）
    2. 各板块周期位置评估
    3. 本周操作是否符合三大纪律的自检清单
    """
    # 先生成每日报告
    daily_report()

    _print_header("每周加强报告")

    # 1. 持仓汇总表
    _print_section("持仓汇总（按板块）")

    if _has_positions():
        prices = _get_realtime_prices()
        total_value = _get_portfolio_total_value(STOCK_POOL, prices)

        if total_value > 0:
            # 按板块汇总
            sector_data = {}
            for code, info in STOCK_POOL.items():
                if info["entry_price"] is None or info["shares"] <= 0:
                    continue
                sector = info["sector"]
                if sector not in sector_data:
                    sector_data[sector] = {"stocks": [], "total_cost": 0, "total_value": 0}

                current_price = prices.get(code, info["entry_price"])
                cost = info["entry_price"] * info["shares"]
                value = current_price * info["shares"]
                pnl_pct = (current_price - info["entry_price"]) / info["entry_price"] * 100

                sector_data[sector]["stocks"].append({
                    "code": code,
                    "name": info["name"],
                    "cost": cost,
                    "value": value,
                    "pnl_pct": pnl_pct,
                })
                sector_data[sector]["total_cost"] += cost
                sector_data[sector]["total_value"] += value

            for sector, data in sector_data.items():
                sector_pct = data["total_value"] / total_value * 100
                sector_pnl = (data["total_value"] - data["total_cost"]) / data["total_cost"] * 100
                status = "正常" if sector_pct <= MAX_SECTOR_PCT * 100 else "⚠️ 超配"

                print(f"\n  【{sector}】仓位占比: {sector_pct:.1f}%  浮盈亏: {_pct_str(sector_pnl)}  {status}")
                for s in data["stocks"]:
                    print(f"    {s['name']}({s['code']}) 浮盈亏: {_pct_str(s['pnl_pct'])}")
        else:
            print("  总市值为零，无法计算仓位比例")
    else:
        print("  当前无持仓")

    # 2. 各板块周期位置评估（基于期货数据）
    _print_section("板块周期位置参考")
    print("  （基于对应商品期货近 3 个月走势）")

    sector_futures_map = {
        "铜": "CU0",
        "铝": "AL0",
        "石油": "SC0",
        "煤化工": "MA0",
    }

    for sector, futures_code in sector_futures_map.items():
        df = get_futures_data(futures_code, days=90)
        if df is None or len(df) < 5:
            print(f"  {sector}: 数据不足，无法评估")
            continue

        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])
        if df.empty:
            continue

        latest = df["close"].iloc[-1]
        quarter_pct = (latest - df["close"].iloc[0]) / df["close"].iloc[0] * 100
        quarter_high = df["close"].max()
        quarter_low = df["close"].min()

        # 当前价格在区间中的位置
        if quarter_high != quarter_low:
            position_in_range = (latest - quarter_low) / (quarter_high - quarter_low) * 100
        else:
            position_in_range = 50

        trend = _trend_label(quarter_pct)
        print(
            f"  {sector}: 季度涨跌 {_pct_str(quarter_pct)}, "
            f"区间位置 {position_in_range:.0f}%, 趋势: {trend}"
        )
        time.sleep(0.3)

    # 3. 三大纪律自检清单
    _print_section("三大纪律周度自检")
    print()
    print("  第一条：严格的风险管理")
    print("    [ ] 本周所有交易的单笔亏损是否控制在总资本 2% 以内？")
    print("    [ ] 所有持仓是否都有明确的止损位？")
    print("    [ ] 个股仓位是否都在 20% 以下？板块仓位是否在 40% 以下？")
    print()
    print("  第二条：无条件执行交易计划")
    print("    [ ] 本周是否有计划外的冲动交易？")
    print("    [ ] 触发的信号是否都被执行？是否有犹豫不决的情况？")
    print("    [ ] 是否有因为'感觉'而修改止损/止盈位？")
    print()
    print("  第三条：保持一致性")
    print("    [ ] 是否在一致地使用同一套交易系统？")
    print("    [ ] 每一笔交易是否都记录在案？")
    print("    [ ] 是否因为一两次亏损就想换策略？")

    print()
    print(f"周报生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * LINE_WIDTH)


# ========== 单只股票详情 ==========

def stock_detail(stock_code: str) -> None:
    """查看单只股票的详细信息

    Args:
        stock_code: 股票代码
    """
    info = STOCK_POOL.get(stock_code, {"name": stock_code, "sector": "未知"})
    name = info.get("name", stock_code)
    sector = info.get("sector", "未知")

    _print_header(f"股票详情: {name} ({stock_code}) — {sector}板块")

    # 1. 日线数据概要
    _print_section("近期行情")
    df = get_stock_daily(stock_code, days=30)
    if df is not None and not df.empty:
        latest = df.iloc[-1]
        print(f"  最新日期: {latest.get('date', '-')}")
        print(f"  收盘价: {latest.get('close', '-')}")
        print(f"  最高: {latest.get('high', '-')}  最低: {latest.get('low', '-')}")

        # 近期走势
        if len(df) >= 5:
            close_5 = df["close"].iloc[-5]
            close_now = df["close"].iloc[-1]
            pct_5d = (close_now - close_5) / close_5 * 100
            print(f"  近 5 日涨跌: {_pct_str(pct_5d)}")

        if len(df) >= 20:
            close_20 = df["close"].iloc[-20]
            close_now = df["close"].iloc[-1]
            pct_20d = (close_now - close_20) / close_20 * 100
            print(f"  近 20 日涨跌: {_pct_str(pct_20d)}")

        # 近 30 日区间
        high_30 = df["high"].max()
        low_30 = df["low"].min()
        print(f"  近 30 日最高: {high_30}  最低: {low_30}")
    else:
        print("  日线数据获取失败")

    # 2. 财务指标
    _print_section("财务指标")
    fin = get_stock_financials(stock_code)
    if fin:
        for key, label in [
            ("eps", "每股收益"),
            ("roe", "加权 ROE(%)"),
            ("bvps", "每股净资产"),
            ("current_ratio", "流动比率"),
            ("quick_ratio", "速动比率"),
        ]:
            val = fin.get(key)
            if val is not None:
                print(f"  {label}: {val}")
    else:
        print("  财务数据获取失败")

    # 3. 分红历史
    _print_section("分红历史（近 5 年）")
    div_df = get_dividend_history(stock_code)
    if div_df is not None and not div_df.empty:
        # 只显示最近 5 条
        for _, row in div_df.head(5).iterrows():
            print(f"  {row.to_dict()}")
    else:
        print("  分红数据获取失败或无分红记录")

    # 4. 资金流向
    _print_section("近期资金流向")
    flow_df = get_fund_flow(stock_code)
    if flow_df is not None and not flow_df.empty:
        # 显示最近 5 个交易日
        for _, row in flow_df.head(5).iterrows():
            print(f"  {row.to_dict()}")
    else:
        print("  资金流向数据获取失败")

    # 5. 持仓信息
    if info.get("entry_price") is not None and info.get("shares", 0) > 0:
        _print_section("持仓信息")
        entry = info["entry_price"]
        shares = info["shares"]
        if df is not None and not df.empty:
            current = df["close"].iloc[-1]
            pnl = (current - entry) * shares
            pnl_pct = (current - entry) / entry * 100
            print(f"  买入均价: {entry:.2f}")
            print(f"  持仓股数: {shares}")
            print(f"  当前收盘价: {current:.2f}")
            print(f"  浮盈亏: {pnl:+.2f} ({_pct_str(pnl_pct)})")

    print()
    print(f"详情生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * LINE_WIDTH)


# ========== 标准数据接口 ==========

def get_daily_data(stock_code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """标准数据接口 — 供用户自有技术指标使用

    获取指定日期范围内的日线 OHLCV 数据（前复权）。
    这是用户技术指标系统的标准输入格式。

    Args:
        stock_code: 股票代码，如 "601899"
        start_date: 开始日期，如 "2025-01-01"
        end_date: 结束日期，如 "2025-12-31"

    Returns:
        DataFrame，列: date, open, high, low, close, volume
        如果获取失败返回 None

    示例:
        df = get_daily_data("601899", "2025-01-01", "2025-06-30")
        # 然后将 df 传入你的技术指标计算函数
    """
    # 将日期格式统一为 YYYYMMDD
    start_fmt = start_date.replace("-", "")
    end_fmt = end_date.replace("-", "")

    df = _safe_call(
        ak.stock_zh_a_hist,
        symbol=stock_code,
        period="daily",
        start_date=start_fmt,
        end_date=end_fmt,
        adjust="qfq",
    )
    if df is None or df.empty:
        return None

    # 统一列名为标准 OHLCV 格式
    df = df.rename(columns={
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
    })

    # 只保留标准列
    standard_cols = ["date", "open", "high", "low", "close", "volume"]
    available_cols = [c for c in standard_cols if c in df.columns]
    df = df[available_cols].reset_index(drop=True)

    return df


# ========== CLI 入口 ==========

def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="周期股监控系统 — 基于 MR Dang 投资哲学",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python cyclical_monitor.py                          # 默认生成每日报告
  python cyclical_monitor.py --daily                  # 每日报告
  python cyclical_monitor.py --weekly                 # 每周报告（含三大纪律自检）
  python cyclical_monitor.py --futures                # 商品期货仪表盘
  python cyclical_monitor.py --stock 601899           # 查看紫金矿业详情
  python cyclical_monitor.py --data 601899 2025-01-01 2025-12-31  # 导出数据

说明:
  - 持仓信息请在脚本顶部 STOCK_POOL 中配置 entry_price 和 shares
  - 期货品种可在 FUTURES_CODES 中增减
  - 仓位参数可在配置区域调整
        """,
    )
    parser.add_argument("--daily", action="store_true", help="生成每日报告")
    parser.add_argument("--weekly", action="store_true", help="生成每周报告（含三大纪律自检）")
    parser.add_argument("--stock", type=str, help="查看单只股票详情，如 601899")
    parser.add_argument("--futures", action="store_true", help="查看商品期货仪表盘")
    parser.add_argument(
        "--data",
        type=str,
        nargs=3,
        metavar=("CODE", "START", "END"),
        help="获取股票数据，如 601899 2025-01-01 2025-12-31",
    )
    args = parser.parse_args()

    if args.daily:
        daily_report()
    elif args.weekly:
        weekly_report()
    elif args.futures:
        commodity_cycle_dashboard()
    elif args.stock:
        stock_detail(args.stock)
    elif args.data:
        code, start, end = args.data
        name = STOCK_POOL.get(code, {}).get("name", code)
        print(f"正在获取 {name}({code}) 从 {start} 到 {end} 的日线数据...")
        df = get_daily_data(code, start, end)
        if df is not None:
            print(f"\n共获取 {len(df)} 条记录:")
            print(df.to_string(index=False))
            # 同时输出 CSV 方便外部使用
            csv_file = f"{code}_{start}_{end}.csv"
            df.to_csv(csv_file, index=False)
            print(f"\n数据已保存至: {csv_file}")
        else:
            print("[错误] 数据获取失败，请检查股票代码和日期范围")
            sys.exit(1)
    else:
        # 默认：每日报告
        daily_report()


if __name__ == "__main__":
    main()
