#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
周期股金字塔建仓策略回测 — 基于 backtrader + 本地不复权数据
=====================================================

用途：用历史数据验证金字塔建仓策略的表现，给你信心或暴露问题。
定位：纯回测，不碰交易接口。你看信号，手动操作。
数据：使用 stock_data_bfq/ 目录下的本地不复权 CSV 数据，完全离线运行。

策略逻辑（来自 MR Dang 投资方案 + 金字塔建仓）：
  - 第1层: 初始建仓 (总资金的 LAYER1_PCT)
  - 第2层: 从建仓价跌 DROP_TRIGGER_PCT，加仓 (LAYER2_PCT)
  - 第3层: 从建仓价跌 2×DROP_TRIGGER_PCT，加仓 (LAYER3_PCT)
  - 第4层: 从建仓价跌 3×DROP_TRIGGER_PCT，满仓 (LAYER4_PCT)
  - 止盈: 从均价涨 TAKE_PROFIT_PCT → 全部卖出
  - 止损: 从均价跌 STOP_LOSS_PCT → 全部卖出
  - 冷却: 止损后等 COOLDOWN_BARS 个交易日再允许重新建仓

用法：
  # 单只股票回测（默认紫金矿业，全部数据）
  python cyclical_backtest.py

  # 指定股票和时间段
  python cyclical_backtest.py --stock 601919 --start 2020-01-01 --end 2025-12-31

  # 多只股票组合回测
  python cyclical_backtest.py --portfolio

  # 调参数
  python cyclical_backtest.py --stock 601899 --drop 0.08 --profit 0.25 --loss 0.15

  # 出图（需要 matplotlib）
  python cyclical_backtest.py --stock 601899 --plot

  # 扫描所有本地数据股票进行回测
  python cyclical_backtest.py --scan --start 2020-01-01

  # 扫描并按收益率排序，只显示 Top N
  python cyclical_backtest.py --scan --top 20

依赖：
  pip install backtrader pandas matplotlib
"""

import argparse
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

import backtrader as bt
import backtrader.analyzers as btanalyzers
import pandas as pd

# ============================================================================
# 本地数据目录
# ============================================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_data_bfq")


def load_local_data(stock_code: str, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
    """
    从 stock_data_bfq/ 目录加载本地不复权 CSV 数据。

    CSV 格式: date,股票代码,open,close,high,low,volume,成交额,振幅,涨跌幅,涨跌额,换手率

    返回 backtrader 需要的 DataFrame (index=date, columns=open/high/low/close/volume/openinterest)
    """
    csv_path = os.path.join(DATA_DIR, f"{stock_code}.csv")
    if not os.path.exists(csv_path):
        print(f"  [错误] 本地数据不存在: {csv_path}")
        return None

    try:
        df = pd.read_csv(csv_path, parse_dates=["date"])
    except Exception as e:
        print(f"  [错误] 读取 {csv_path} 失败: {e}")
        return None

    if df.empty:
        print(f"  [警告] {stock_code} 数据文件为空")
        return None

    # 按日期排序
    df.sort_values("date", inplace=True)

    # 日期筛选
    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)]

    if df.empty:
        print(f"  [警告] {stock_code} 在指定日期范围内无数据")
        return None

    # 设置日期为索引
    df.set_index("date", inplace=True)

    # 提取 backtrader 需要的列（不复权价格直接用）
    result = df[["open", "high", "low", "close", "volume"]].copy()
    result["openinterest"] = 0

    # 确保数值类型
    for col in ["open", "high", "low", "close", "volume"]:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    result.dropna(subset=["open", "high", "low", "close"], inplace=True)

    if result.empty:
        print(f"  [警告] {stock_code} 清洗后无有效数据")
        return None

    return result


def list_local_stocks() -> List[str]:
    """列出所有有本地数据的股票代码"""
    if not os.path.isdir(DATA_DIR):
        return []
    codes = []
    for f in sorted(os.listdir(DATA_DIR)):
        if f.endswith(".csv"):
            codes.append(f.replace(".csv", ""))
    return codes


# ============================================================================
# 金字塔建仓策略
# ============================================================================

class PyramidStrategy(bt.Strategy):
    """
    金字塔建仓策略

    原理：
      跌得越多，买得越多（前提是龙头 + 周期底部区域）
      涨到目标，分批止盈（这里简化为一次性止盈）

    注意：使用不复权数据，价格是原始交易价格。
    """

    params = dict(
        # 各层仓位占总资金的比例
        layer1_pct=0.10,      # 第1层: 试探性建仓 10%
        layer2_pct=0.15,      # 第2层: 确认下跌 15%
        layer3_pct=0.25,      # 第3层: 深度回调 25%
        layer4_pct=0.30,      # 第4层: 满仓抄底 30% (累计 80%)

        # 补仓触发跌幅（从首次建仓价算）
        drop_trigger_pct=0.10,  # 每跌 10% 加一层

        # 止盈止损（从持仓均价算）
        take_profit_pct=0.30,   # 盈利 30% 止盈
        stop_loss_pct=0.99,     # 亏损 20% 止损

        # 冷却期: 止损后等多少个交易日才能重新建仓
        cooldown_bars=20,

        # 是否打印交易日志
        printlog=True,

        # 股票名称（用于日志）
        stock_name="",
    )

    def __init__(self):
        self.dataclose = self.datas[0].close

        # 状态追踪
        self.order = None           # 当前挂单
        self.first_entry_price = 0  # 首次建仓价格
        self.current_layer = 0     # 当前层数 (0=空仓, 1-4)
        self.avg_cost = 0          # 持仓均价
        self.total_invested = 0    # 累计投入金额
        self.total_shares = 0      # 累计持仓股数
        self.last_invested = 0     # 上次平仓前的投入金额（用于计算收益率）
        self.cooldown_count = 0    # 冷却期计数器

        # 交易记录
        self.trade_log: List[dict] = []

    def log(self, txt, dt=None, force=False):
        if self.p.printlog or force:
            dt = dt or self.datas[0].datetime.date(0)
            name = f"[{self.p.stock_name}] " if self.p.stock_name else ""
            print(f"  {dt.isoformat()} {name}{txt}")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"买入成交 价格:{order.executed.price:.2f} "
                         f"数量:{order.executed.size:.0f} "
                         f"手续费:{order.executed.comm:.2f}")
                # 更新均价
                cost = order.executed.price * order.executed.size + order.executed.comm
                self.total_invested += cost
                self.total_shares += order.executed.size
                if self.total_shares > 0:
                    self.avg_cost = self.total_invested / self.total_shares
            else:
                self.log(f"卖出成交 价格:{order.executed.price:.2f} "
                         f"数量:{abs(order.executed.size):.0f} "
                         f"手续费:{order.executed.comm:.2f}")

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("订单被取消/保证金不足/被拒绝")

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        pnl_pct = (trade.pnlcomm / self.last_invested) * 100 if self.last_invested > 0 else 0
        self.log(f"交易结束 毛利:{trade.pnl:.2f} 净利:{trade.pnlcomm:.2f} "
                 f"收益率:{pnl_pct:.1f}%", force=True)

        self.trade_log.append({
            "date": self.datas[0].datetime.date(0).isoformat(),
            "pnl": round(trade.pnlcomm, 2),
            "pnl_pct": round(pnl_pct, 1),
        })

    def _calc_shares(self, pct: float) -> int:
        """根据百分比计算可买股数（A股最少100股/手）"""
        total_value = self.broker.getvalue()
        target_value = total_value * pct
        price = self.dataclose[0]
        if price <= 0:
            return 0
        shares = int(target_value / price)
        # A 股最少买 100 股（1手）
        shares = (shares // 100) * 100
        return max(shares, 0)

    def _reset_position_state(self):
        """重置仓位状态"""
        self.last_invested = self.total_invested
        self.first_entry_price = 0
        self.current_layer = 0
        self.avg_cost = 0
        self.total_invested = 0
        self.total_shares = 0

    def next(self):
        # 有挂单就等着
        if self.order:
            return

        # 冷却期计数
        if self.cooldown_count > 0:
            self.cooldown_count -= 1
            return

        current_price = self.dataclose[0]
        position_size = self.position.size

        # ==================== 持仓中 ====================
        if position_size > 0 and self.current_layer > 0:
            # 止盈检查: 当前价 vs 均价
            if self.avg_cost > 0:
                gain_pct = (current_price - self.avg_cost) / self.avg_cost

                # 🎯 触发止盈
                if gain_pct >= self.p.take_profit_pct:
                    self.log(f"🎯 止盈! 均价:{self.avg_cost:.2f} 现价:{current_price:.2f} "
                             f"盈利:{gain_pct*100:.1f}%", force=True)
                    self.order = self.close()  # 全部卖出
                    self._reset_position_state()
                    return

                # 🚨 触发止损
                if gain_pct <= -self.p.stop_loss_pct:
                    self.log(f"🚨 止损! 均价:{self.avg_cost:.2f} 现价:{current_price:.2f} "
                             f"亏损:{gain_pct*100:.1f}%", force=True)
                    self.order = self.close()  # 全部卖出
                    self._reset_position_state()
                    self.cooldown_count = self.p.cooldown_bars  # 进入冷却期
                    return

            # 补仓检查: 当前价 vs 首次建仓价
            if self.first_entry_price > 0:
                drop_from_entry = (current_price - self.first_entry_price) / self.first_entry_price

                # 第2层: 跌 1×drop_trigger
                if self.current_layer == 1 and drop_from_entry <= -self.p.drop_trigger_pct:
                    shares = self._calc_shares(self.p.layer2_pct)
                    if shares > 0:
                        self.log(f"⚠️ 补仓第2层 跌幅:{drop_from_entry*100:.1f}% "
                                 f"加仓{shares}股")
                        self.order = self.buy(size=shares)
                        self.current_layer = 2
                    return

                # 第3层: 跌 2×drop_trigger
                if self.current_layer == 2 and drop_from_entry <= -self.p.drop_trigger_pct * 2:
                    shares = self._calc_shares(self.p.layer3_pct)
                    if shares > 0:
                        self.log(f"⚠️ 补仓第3层 跌幅:{drop_from_entry*100:.1f}% "
                                 f"加仓{shares}股")
                        self.order = self.buy(size=shares)
                        self.current_layer = 3
                    return

                # 第4层: 跌 3×drop_trigger（满仓）
                if self.current_layer == 3 and drop_from_entry <= -self.p.drop_trigger_pct * 3:
                    shares = self._calc_shares(self.p.layer4_pct)
                    if shares > 0:
                        self.log(f"⚠️ 满仓第4层 跌幅:{drop_from_entry*100:.1f}% "
                                 f"加仓{shares}股")
                        self.order = self.buy(size=shares)
                        self.current_layer = 4
                    return

        # ==================== 空仓 → 寻找建仓机会 ====================
        elif position_size == 0 and self.current_layer == 0:
            # 简单入场条件：价格从近期高点回落一定幅度后建仓
            # （实际使用时，这里应该结合你的技术指标判断周期底部）
            #
            # 回测中我们用一个简化逻辑：
            #   - 如果最近 60 个交易日的最低价就是今天或近 5 天内 → 认为在底部区域
            #   - 且价格从 60 日高点已经回落超过 15%
            #
            # 你可以替换成自己的技术指标判断

            if len(self) < 60:
                return  # 数据不够，等着

            high_60 = max(self.data.high.get(size=60))
            low_5 = min(self.data.low.get(size=5))

            # 从 60 日高点的回撤幅度
            drawdown = (current_price - high_60) / high_60 if high_60 > 0 else 0

            # 条件: 回撤超过 15%，且近 5 日触及 60 日低点附近
            low_60 = min(self.data.low.get(size=60))
            near_low = low_5 <= low_60 * 1.03  # 近 5 日最低价在 60 日最低价的 3% 范围内

            if drawdown <= -0.15 and near_low:
                shares = self._calc_shares(self.p.layer1_pct)
                if shares >= 100:
                    self.log(f"📈 首次建仓 价格:{current_price:.2f} "
                             f"60日回撤:{drawdown*100:.1f}% 买入{shares}股")
                    self.order = self.buy(size=shares)
                    self.first_entry_price = current_price
                    self.current_layer = 1


# ============================================================================
# 组合策略（多只股票同时回测）
# ============================================================================

class PortfolioPyramidStrategy(bt.Strategy):
    """
    多股票组合金字塔策略

    对每只股票独立执行金字塔逻辑，同时控制：
      - 单只股票不超过总资金 20%
      - 总仓位有上限
    """

    params = dict(
        layer1_pct=0.05,        # 组合模式下每只首次建仓更小
        layer2_pct=0.05,
        layer3_pct=0.05,
        layer4_pct=0.05,
        drop_trigger_pct=0.10,
        take_profit_pct=0.30,
        stop_loss_pct=0.20,
        max_single_pct=0.20,   # 单只不超过 20%
        cooldown_bars=20,
        printlog=True,
    )

    def __init__(self):
        self.orders = {}
        self.stock_state = {}

        for i, d in enumerate(self.datas):
            name = d._name
            self.stock_state[name] = {
                "first_entry_price": 0,
                "current_layer": 0,
                "avg_cost": 0,
                "total_invested": 0,
                "total_shares": 0,
                "cooldown": 0,
            }
            self.orders[name] = None

    def log(self, txt, dt=None, force=False):
        if self.p.printlog or force:
            dt = dt or self.datas[0].datetime.date(0)
            print(f"  {dt.isoformat()} {txt}")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        name = order.data._name
        if order.status in [order.Completed]:
            state = self.stock_state[name]
            if order.isbuy():
                cost = order.executed.price * order.executed.size + order.executed.comm
                state["total_invested"] += cost
                state["total_shares"] += order.executed.size
                if state["total_shares"] > 0:
                    state["avg_cost"] = state["total_invested"] / state["total_shares"]
                self.log(f"[{name}] 买入 价格:{order.executed.price:.2f} 数量:{order.executed.size:.0f}")
            else:
                self.log(f"[{name}] 卖出 价格:{order.executed.price:.2f} 数量:{abs(order.executed.size):.0f}")

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"[{name}] 订单取消/拒绝")

        self.orders[name] = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        name = trade.data._name
        self.log(f"[{name}] 交易结束 净利:{trade.pnlcomm:.2f}", force=True)

    def _calc_shares(self, pct, price):
        total_value = self.broker.getvalue()
        target_value = total_value * pct
        if price <= 0:
            return 0
        shares = int(target_value / price)
        return (shares // 100) * 100

    def _check_single_limit(self, data_name, add_value):
        """检查单只股票是否超过仓位上限"""
        total_value = self.broker.getvalue()
        # 当前持仓市值
        for d in self.datas:
            if d._name == data_name:
                pos = self.getposition(d)
                current_value = pos.size * d.close[0] if pos.size > 0 else 0
                if (current_value + add_value) / total_value > self.p.max_single_pct:
                    return False
                return True
        return True

    def _reset_state(self, name):
        self.stock_state[name] = {
            "first_entry_price": 0,
            "current_layer": 0,
            "avg_cost": 0,
            "total_invested": 0,
            "total_shares": 0,
            "cooldown": self.stock_state[name].get("cooldown", 0),
        }

    def next(self):
        for i, d in enumerate(self.datas):
            name = d._name
            state = self.stock_state[name]

            if self.orders[name]:
                continue

            if state["cooldown"] > 0:
                state["cooldown"] -= 1
                continue

            price = d.close[0]
            pos = self.getposition(d)

            # 持仓逻辑
            if pos.size > 0 and state["current_layer"] > 0:
                avg = state["avg_cost"]
                if avg > 0:
                    gain = (price - avg) / avg

                    if gain >= self.p.take_profit_pct:
                        self.log(f"[{name}] 🎯 止盈 盈利:{gain*100:.1f}%", force=True)
                        self.orders[name] = self.close(data=d)
                        self._reset_state(name)
                        continue

                    if gain <= -self.p.stop_loss_pct:
                        self.log(f"[{name}] 🚨 止损 亏损:{gain*100:.1f}%", force=True)
                        self.orders[name] = self.close(data=d)
                        self._reset_state(name)
                        state["cooldown"] = self.p.cooldown_bars
                        continue

                if state["first_entry_price"] > 0:
                    drop = (price - state["first_entry_price"]) / state["first_entry_price"]
                    layer = state["current_layer"]
                    layer_map = {
                        1: (self.p.drop_trigger_pct, self.p.layer2_pct, 2),
                        2: (self.p.drop_trigger_pct * 2, self.p.layer3_pct, 3),
                        3: (self.p.drop_trigger_pct * 3, self.p.layer4_pct, 4),
                    }
                    if layer in layer_map:
                        threshold, pct, next_layer = layer_map[layer]
                        if drop <= -threshold:
                            shares = self._calc_shares(pct, price)
                            add_value = shares * price
                            if shares >= 100 and self._check_single_limit(name, add_value):
                                self.log(f"[{name}] 补仓第{next_layer}层 跌幅:{drop*100:.1f}%")
                                self.orders[name] = self.buy(data=d, size=shares)
                                state["current_layer"] = next_layer

            # 空仓 → 建仓
            elif pos.size == 0 and state["current_layer"] == 0:
                if len(self) < 60:
                    continue

                highs = [d.high[-j] for j in range(60) if j < len(self)]
                lows_5 = [d.low[-j] for j in range(5) if j < len(self)]
                lows_60 = [d.low[-j] for j in range(60) if j < len(self)]

                if not highs or not lows_5 or not lows_60:
                    continue

                high_60 = max(highs)
                low_5 = min(lows_5)
                low_60 = min(lows_60)

                drawdown = (price - high_60) / high_60 if high_60 > 0 else 0
                near_low = low_5 <= low_60 * 1.03

                if drawdown <= -0.15 and near_low:
                    shares = self._calc_shares(self.p.layer1_pct, price)
                    add_value = shares * price
                    if shares >= 100 and self._check_single_limit(name, add_value):
                        self.log(f"[{name}] 📈 建仓 回撤:{drawdown*100:.1f}%")
                        self.orders[name] = self.buy(data=d, size=shares)
                        state["first_entry_price"] = price
                        state["current_layer"] = 1


# ============================================================================
# 回测引擎
# ============================================================================

# 默认股票池（与 cyclical_monitor.py 保持一致）
DEFAULT_STOCKS = {
    "601899": "紫金矿业",
    "601168": "西部矿业",
    "600362": "江西铜业",
    "600219": "南山铝业",
    "000933": "神火股份",
    "600989": "宝丰能源",
    "000792": "盐湖股份",
    "601919": "中远海控",
    "600938": "中国海油",
}


def run_single_backtest(
    stock_code: str,
    stock_name: str,
    start_date: str = None,
    end_date: str = None,
    initial_cash: float = 100_000,
    drop_trigger: float = 0.10,
    take_profit: float = 0.30,
    stop_loss: float = 0.20,
    do_plot: bool = False,
    printlog: bool = True,
) -> Optional[dict]:
    """运行单只股票回测（使用本地不复权数据）"""

    print(f"\n{'='*60}")
    print(f"  回测: {stock_name} ({stock_code})")
    if start_date or end_date:
        print(f"  时段: {start_date or '最早'} ~ {end_date or '最新'}")
    print(f"  初始资金: {initial_cash:,.0f} 元")
    print(f"  数据类型: 不复权")
    print(f"  参数: 补仓跌幅={drop_trigger*100:.0f}% 止盈={take_profit*100:.0f}% 止损={stop_loss*100:.0f}%")
    print(f"{'='*60}")

    # 加载本地数据
    df = load_local_data(stock_code, start_date, end_date)
    if df is None:
        return None

    print(f"  数据: {len(df)} 个交易日 ({df.index[0].date()} ~ {df.index[-1].date()})")

    # 构建引擎
    cerebro = bt.Cerebro()

    # 加载数据
    data_feed = bt.feeds.PandasData(dataname=df, name=stock_name)
    cerebro.adddata(data_feed)

    # 加载策略
    cerebro.addstrategy(
        PyramidStrategy,
        drop_trigger_pct=drop_trigger,
        take_profit_pct=take_profit,
        stop_loss_pct=stop_loss,
        printlog=printlog,
        stock_name=stock_name,
    )

    # 资金和手续费
    cerebro.broker.setcash(initial_cash)
    # A 股手续费: 万2.5 佣金 + 千1 印花税(卖出)
    cerebro.broker.setcommission(
        commission=0.00025,  # 万2.5
        stocklike=True,
    )

    # 分析器
    cerebro.addanalyzer(btanalyzers.SharpeRatio, _name="sharpe", riskfreerate=0.02)
    cerebro.addanalyzer(btanalyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(btanalyzers.Returns, _name="returns")
    cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name="trades")

    # 运行
    start_value = cerebro.broker.getvalue()
    results = cerebro.run()
    end_value = cerebro.broker.getvalue()
    strat = results[0]

    # 输出结果
    total_return = (end_value - start_value) / start_value * 100
    print(f"\n{'—'*60}")
    print(f"  回测结果: {stock_name}")
    print(f"{'—'*60}")
    print(f"  期初资金:    {start_value:>12,.2f} 元")
    print(f"  期末资金:    {end_value:>12,.2f} 元")
    print(f"  总收益率:    {total_return:>11.2f}%")

    # 年化收益
    returns_analysis = strat.analyzers.returns.get_analysis()
    annual_return = returns_analysis.get("rnorm100", 0)
    print(f"  年化收益:    {annual_return:>11.2f}%")

    # 最大回撤
    dd = strat.analyzers.drawdown.get_analysis()
    max_dd = dd.get("max", {}).get("drawdown", 0)
    print(f"  最大回撤:    {max_dd:>11.2f}%")

    # 夏普比率
    sharpe = strat.analyzers.sharpe.get_analysis()
    sharpe_ratio = sharpe.get("sharperatio", None)
    sharpe_str = f"{sharpe_ratio:.2f}" if sharpe_ratio else "N/A"
    print(f"  夏普比率:    {sharpe_str:>11}")

    # 交易统计
    trade_analysis = strat.analyzers.trades.get_analysis()
    total_trades = trade_analysis.get("total", {}).get("total", 0)
    won = trade_analysis.get("won", {}).get("total", 0)
    lost = trade_analysis.get("lost", {}).get("total", 0)
    win_rate = (won / total_trades * 100) if total_trades > 0 else 0
    print(f"  总交易次数:  {total_trades:>11}")
    print(f"  盈利次数:    {won:>11}")
    print(f"  亏损次数:    {lost:>11}")
    print(f"  胜率:        {win_rate:>10.1f}%")

    # 交易明细
    if strat.trade_log:
        print(f"\n  交易明细:")
        for t in strat.trade_log:
            emoji = "✅" if t["pnl"] >= 0 else "❌"
            print(f"    {emoji} {t['date']}  净利:{t['pnl']:>+10,.2f}  收益率:{t['pnl_pct']:>+6.1f}%")

    print(f"{'='*60}\n")

    # 画图
    if do_plot:
        try:
            cerebro.plot(
                style="candle",
                barup="red", bardown="green",  # A 股颜色习惯
                volup="red", voldown="green",
            )
        except Exception as e:
            print(f"  [提示] 画图失败 (可能没有图形界面): {e}")

    return {
        "stock_code": stock_code,
        "stock_name": stock_name,
        "total_return": round(total_return, 2),
        "annual_return": round(annual_return, 2),
        "max_drawdown": round(max_dd, 2),
        "sharpe_ratio": round(sharpe_ratio, 2) if sharpe_ratio else None,
        "total_trades": total_trades,
        "win_rate": round(win_rate, 1),
    }


def run_portfolio_backtest(
    start_date: str = None,
    end_date: str = None,
    initial_cash: float = 500_000,
    drop_trigger: float = 0.10,
    take_profit: float = 0.30,
    stop_loss: float = 0.20,
    do_plot: bool = False,
) -> Optional[dict]:
    """运行多股票组合回测（使用本地不复权数据）"""

    print(f"\n{'█'*60}")
    print(f"  组合回测: {len(DEFAULT_STOCKS)} 只周期龙头股")
    if start_date or end_date:
        print(f"  时段: {start_date or '最早'} ~ {end_date or '最新'}")
    print(f"  初始资金: {initial_cash:,.0f} 元")
    print(f"  数据类型: 不复权")
    print(f"{'█'*60}")

    cerebro = bt.Cerebro()

    # 加载所有股票数据
    loaded = 0
    for code, name in DEFAULT_STOCKS.items():
        df = load_local_data(code, start_date, end_date)
        if df is not None and len(df) > 60:
            data_feed = bt.feeds.PandasData(dataname=df, name=name)
            cerebro.adddata(data_feed)
            loaded += 1
            print(f"  ✅ {name}({code}): {len(df)} 交易日")
        else:
            print(f"  ❌ {name}({code}): 数据不足，跳过")

    if loaded == 0:
        print("  [错误] 没有加载到任何有效数据")
        return None

    # 加载策略
    cerebro.addstrategy(
        PortfolioPyramidStrategy,
        drop_trigger_pct=drop_trigger,
        take_profit_pct=take_profit,
        stop_loss_pct=stop_loss,
        printlog=True,
    )

    # 资金和手续费
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=0.00025, stocklike=True)

    # 分析器
    cerebro.addanalyzer(btanalyzers.SharpeRatio, _name="sharpe", riskfreerate=0.02)
    cerebro.addanalyzer(btanalyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(btanalyzers.Returns, _name="returns")
    cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name="trades")

    # 运行
    start_value = cerebro.broker.getvalue()
    results = cerebro.run()
    end_value = cerebro.broker.getvalue()
    strat = results[0]

    total_return = (end_value - start_value) / start_value * 100
    returns_analysis = strat.analyzers.returns.get_analysis()
    annual_return = returns_analysis.get("rnorm100", 0)
    dd = strat.analyzers.drawdown.get_analysis()
    max_dd = dd.get("max", {}).get("drawdown", 0)
    sharpe = strat.analyzers.sharpe.get_analysis()
    sharpe_ratio = sharpe.get("sharperatio", None)
    trade_analysis = strat.analyzers.trades.get_analysis()
    total_trades = trade_analysis.get("total", {}).get("total", 0)
    won = trade_analysis.get("won", {}).get("total", 0)
    lost = trade_analysis.get("lost", {}).get("total", 0)
    win_rate = (won / total_trades * 100) if total_trades > 0 else 0

    print(f"\n{'█'*60}")
    print(f"  组合回测结果")
    print(f"{'█'*60}")
    print(f"  期初资金:    {start_value:>12,.2f} 元")
    print(f"  期末资金:    {end_value:>12,.2f} 元")
    print(f"  总收益率:    {total_return:>11.2f}%")
    print(f"  年化收益:    {annual_return:>11.2f}%")
    print(f"  最大回撤:    {max_dd:>11.2f}%")
    sharpe_str = f"{sharpe_ratio:.2f}" if sharpe_ratio else "N/A"
    print(f"  夏普比率:    {sharpe_str:>11}")
    print(f"  总交易次数:  {total_trades:>11}")
    print(f"  胜率:        {win_rate:>10.1f}%")
    print(f"{'█'*60}\n")

    if do_plot:
        try:
            cerebro.plot(style="candle", barup="red", bardown="green")
        except Exception as e:
            print(f"  [提示] 画图失败: {e}")

    return {
        "total_return": round(total_return, 2),
        "annual_return": round(annual_return, 2),
        "max_drawdown": round(max_dd, 2),
        "sharpe_ratio": round(sharpe_ratio, 2) if sharpe_ratio else None,
        "total_trades": total_trades,
        "win_rate": round(win_rate, 1),
    }


def run_scan_backtest(
    start_date: str = None,
    end_date: str = None,
    initial_cash: float = 100_000,
    drop_trigger: float = 0.10,
    take_profit: float = 0.30,
    stop_loss: float = 0.20,
    top_n: int = 0,
) -> None:
    """扫描所有本地数据股票进行回测，按收益率排序"""

    all_codes = list_local_stocks()
    print(f"\n{'█'*60}")
    print(f"  全量扫描回测: 共 {len(all_codes)} 只股票")
    if start_date or end_date:
        print(f"  时段: {start_date or '最早'} ~ {end_date or '最新'}")
    print(f"  初始资金: {initial_cash:,.0f} 元")
    print(f"  数据类型: 不复权")
    print(f"{'█'*60}\n")

    results = []
    skipped = 0
    for i, code in enumerate(all_codes):
        # 简单进度
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  进度: {i+1}/{len(all_codes)} ...")

        name = DEFAULT_STOCKS.get(code, code)
        r = run_single_backtest(
            stock_code=code,
            stock_name=name,
            start_date=start_date,
            end_date=end_date,
            initial_cash=initial_cash,
            drop_trigger=drop_trigger,
            take_profit=take_profit,
            stop_loss=stop_loss,
            printlog=False,  # 扫描模式静默
        )
        if r:
            results.append(r)
        else:
            skipped += 1

    # 按总收益率排序
    results.sort(key=lambda x: x["total_return"], reverse=True)

    # 显示结果
    if top_n > 0:
        display = results[:top_n]
        title = f"扫描结果 Top {top_n} (共 {len(results)} 只有效)"
    else:
        display = results
        title = f"扫描结果 (共 {len(results)} 只有效, {skipped} 只跳过)"

    print(f"\n{'█'*60}")
    print(f"  {title}")
    print(f"{'█'*60}")
    print(f"  {'股票':<12} {'代码':<8} {'总收益%':>8} {'年化%':>8} {'回撤%':>8} {'夏普':>6} {'交易数':>6} {'胜率%':>6}")
    print(f"  {'-'*62}")
    for r in display:
        sharpe = f"{r['sharpe_ratio']:.2f}" if r['sharpe_ratio'] else "N/A"
        print(f"  {r['stock_name']:<12} {r['stock_code']:<8} {r['total_return']:>+7.1f} "
              f"{r['annual_return']:>+7.1f} {r['max_drawdown']:>7.1f} {sharpe:>6} "
              f"{r['total_trades']:>6} {r['win_rate']:>5.1f}")
    print(f"{'█'*60}\n")

    # 统计摘要
    if results:
        avg_return = sum(r["total_return"] for r in results) / len(results)
        positive = sum(1 for r in results if r["total_return"] > 0)
        negative = sum(1 for r in results if r["total_return"] < 0)
        zero = sum(1 for r in results if r["total_return"] == 0 and r["total_trades"] == 0)
        print(f"  📊 统计摘要:")
        print(f"     平均收益率: {avg_return:+.2f}%")
        print(f"     盈利股票: {positive} 只  亏损股票: {negative} 只  无交易: {zero} 只")
        if results:
            print(f"     最佳: {results[0]['stock_name']}({results[0]['stock_code']}) {results[0]['total_return']:+.1f}%")
            print(f"     最差: {results[-1]['stock_name']}({results[-1]['stock_code']}) {results[-1]['total_return']:+.1f}%")
        print()


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="周期股金字塔建仓策略回测 — 本地不复权数据 (离线版)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                                              # 默认: 紫金矿业全部数据
  %(prog)s --stock 601919 --start 2020-01-01            # 中远海控
  %(prog)s --portfolio --start 2021-01-01               # 9只股票组合回测
  %(prog)s --stock 601899 --drop 0.08 --profit 0.25     # 调参数
  %(prog)s --stock 601899 --plot                        # 出图
  %(prog)s --all                                        # 逐只回测默认9只龙头股
  %(prog)s --scan --start 2020-01-01                    # 扫描所有本地数据股票
  %(prog)s --scan --top 20                              # 扫描并只显示 Top 20
  %(prog)s --list                                       # 列出所有有本地数据的股票

数据说明:
  使用 stock_data_bfq/ 目录下的不复权 CSV 数据，完全离线运行。
  不依赖 akshare，不需要网络连接。

策略说明:
  金字塔建仓: 第1层(10%) → 跌10%加第2层(15%) → 再跌10%加第3层(25%) → 再跌10%满仓(30%)
  止盈: 从均价涨30%全部卖出
  止损: 从均价跌20%全部卖出 + 冷却20个交易日
        """,
    )
    parser.add_argument("--stock", type=str, default="601899", help="股票代码 (默认: 601899 紫金矿业)")
    parser.add_argument("--start", type=str, default=None, help="回测开始日期 (默认: 全部数据)")
    parser.add_argument("--end", type=str, default=None, help="回测结束日期 (默认: 全部数据)")
    parser.add_argument("--cash", type=float, default=100_000, help="初始资金 (默认: 100000)")
    parser.add_argument("--drop", type=float, default=0.10, help="补仓触发跌幅 (默认: 0.10)")
    parser.add_argument("--profit", type=float, default=0.30, help="止盈线 (默认: 0.30)")
    parser.add_argument("--loss", type=float, default=0.20, help="止损线 (默认: 0.20)")
    parser.add_argument("--plot", action="store_true", help="输出图表")
    parser.add_argument("--portfolio", action="store_true", help="多股票组合回测")
    parser.add_argument("--all", action="store_true", help="逐只回测所有默认股票")
    parser.add_argument("--scan", action="store_true", help="扫描所有本地数据股票进行回测")
    parser.add_argument("--top", type=int, default=0, help="扫描模式下只显示收益率 Top N")
    parser.add_argument("--list", action="store_true", help="列出所有有本地数据的股票代码")
    parser.add_argument("--quiet", action="store_true", help="安静模式（不打印交易日志）")

    args = parser.parse_args()

    printlog = not args.quiet

    if args.list:
        # 列出所有本地数据股票
        codes = list_local_stocks()
        print(f"\n本地数据目录: {DATA_DIR}")
        print(f"共 {len(codes)} 只股票:\n")
        for i, code in enumerate(codes):
            name = DEFAULT_STOCKS.get(code, "")
            label = f"  {code}" + (f"  {name}" if name else "")
            print(label)
        print()

    elif args.scan:
        # 全量扫描回测
        run_scan_backtest(
            start_date=args.start,
            end_date=args.end,
            initial_cash=args.cash,
            drop_trigger=args.drop,
            take_profit=args.profit,
            stop_loss=args.loss,
            top_n=args.top,
        )

    elif args.portfolio:
        # 组合回测
        run_portfolio_backtest(
            start_date=args.start,
            end_date=args.end,
            initial_cash=args.cash,
            drop_trigger=args.drop,
            take_profit=args.profit,
            stop_loss=args.loss,
            do_plot=args.plot,
        )

    elif args.all:
        # 逐只回测默认股票
        results = []
        for code, name in DEFAULT_STOCKS.items():
            r = run_single_backtest(
                stock_code=code,
                stock_name=name,
                start_date=args.start,
                end_date=args.end,
                initial_cash=args.cash,
                drop_trigger=args.drop,
                take_profit=args.profit,
                stop_loss=args.loss,
                printlog=printlog,
            )
            if r:
                results.append(r)

        # 汇总表
        if results:
            print(f"\n{'█'*60}")
            print(f"  逐只回测汇总")
            print(f"{'█'*60}")
            print(f"  {'股票':<10} {'总收益%':>8} {'年化%':>8} {'回撤%':>8} {'夏普':>6} {'胜率%':>6}")
            print(f"  {'-'*50}")
            for r in results:
                sharpe = f"{r['sharpe_ratio']:.2f}" if r['sharpe_ratio'] else "N/A"
                print(f"  {r['stock_name']:<10} {r['total_return']:>+7.1f} {r['annual_return']:>+7.1f} "
                      f"{r['max_drawdown']:>7.1f} {sharpe:>6} {r['win_rate']:>5.1f}")
            print(f"{'█'*60}\n")

    else:
        # 单只回测
        stock_name = DEFAULT_STOCKS.get(args.stock, args.stock)
        run_single_backtest(
            stock_code=args.stock,
            stock_name=stock_name,
            start_date=args.start,
            end_date=args.end,
            initial_cash=args.cash,
            drop_trigger=args.drop,
            take_profit=args.profit,
            stop_loss=args.loss,
            do_plot=args.plot,
            printlog=printlog,
        )


if __name__ == "__main__":
    main()
