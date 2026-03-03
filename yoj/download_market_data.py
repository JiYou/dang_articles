#!/usr/bin/env python3
"""
下载上证指数 + 行业ETF 日线数据 (via akshare / Sina)

使用 stock_zh_index_daily (Sina源), 每次请求间隔20秒
输出到 market_data/ 目录, CSV格式: date,open,high,low,close,volume
"""

import os
import time
import akshare as ak
import pandas as pd

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "market_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 上证指数 + 主要行业ETF + 跨境ETF + 商品ETF
SYMBOLS = [
    # ====== 大盘指数 ======
    ("sh000001", "上证指数"),
    ("sz399001", "深证成指"),
    ("sh000300", "沪深300"),
    ("sz399006", "创业板指"),
    ("sh000016", "上证50"),
    ("sh000905", "中证500"),
    ("sh000852", "中证1000"),
    # ====== 宽基ETF ======
    ("sh510300", "沪深300ETF"),
    ("sh510050", "上证50ETF"),
    ("sh510500", "中证500ETF"),
    ("sz159915", "创业板ETF"),
    ("sh588000", "科创50ETF"),
    # ====== 行业ETF ======
    ("sh512000", "券商ETF"),
    ("sh512800", "银行ETF"),
    ("sz159928", "消费ETF"),
    ("sh512690", "白酒ETF"),
    ("sh512010", "医药ETF"),
    ("sh516160", "新能源ETF"),
    ("sh515790", "光伏ETF"),
    ("sh512480", "半导体ETF"),
    ("sh512660", "军工ETF"),
    ("sh515000", "科技ETF"),
    ("sh512200", "房地产ETF"),
    ("sh515210", "钢铁ETF"),
    ("sh515220", "煤炭ETF"),
    ("sh512400", "有色金属ETF"),
    ("sh512980", "传媒ETF"),
    ("sz159995", "芯片ETF"),
    ("sh512070", "保险ETF"),
    ("sh515170", "食品ETF"),
    ("sh516950", "基建ETF"),
    ("sh515880", "通信ETF"),
    ("sh512170", "医疗ETF"),
    ("sh515030", "新能源车ETF"),
    ("sh562800", "中药ETF"),
    ("sh515650", "稀有金属ETF"),
    ("sh516150", "稀土ETF"),
    ("sh512290", "生物医药ETF"),
    ("sh515710", "人工智能ETF"),
    ("sh512760", "半导体50ETF"),
    ("sh516780", "游戏ETF"),
    ("sh515860", "碳中和ETF"),
    ("sh562510", "旅游ETF"),
    ("sh515080", "中证800ETF"),
    ("sz159870", "化工ETF"),
    ("sz159611", "电力ETF"),
    # ====== 跨境/海外ETF ======
    ("sh513100", "纳指ETF"),
    ("sz159941", "纳指100ETF"),
    ("sh513060", "恒生医疗ETF"),
    ("sz159920", "恒生ETF"),
    ("sh513050", "中概互联ETF"),
    ("sh513030", "德国ETF"),
    ("sh513080", "法国ETF"),
    ("sh513520", "日经ETF"),
    ("sh513000", "标普ETF"),
    ("sh513880", "日经225ETF"),
    ("sz159866", "日本东证ETF"),
    ("sh513090", "港股通科技ETF"),
    ("sh513010", "恒生科技ETF"),
    ("sh513180", "恒生红利ETF"),
    ("sh513330", "美国50ETF"),
    ("sh513550", "标普油气ETF"),
    ("sz159607", "亚太精选ETF"),
    # ====== 商品/黄金ETF ======
    ("sh518880", "黄金ETF"),
    ("sz159934", "黄金ETF基金"),
    ("sh518800", "黄金股ETF"),
    ("sz159981", "能源化工ETF"),
    ("sz159985", "豆粕ETF"),
    ("sh518660", "黄金ETF龙头"),
    ("sz159869", "有色50ETF"),
    # ====== 红利/价值/成长 风格ETF ======
    ("sh510880", "红利ETF"),
    ("sh515180", "红利100ETF"),
    ("sh512890", "红利低波ETF"),
    ("sh510090", "央企ETF"),
    ("sh561990", "A50ETF"),
]

DELAY_SECONDS = 20


def download_one(symbol: str, name: str) -> bool:
    """下载单个品种, 返回是否成功"""
    out_path = os.path.join(OUTPUT_DIR, f"{symbol}.csv")
    if os.path.exists(out_path):
        df_existing = pd.read_csv(out_path)
        print(f"  [SKIP] {name} ({symbol}) already exists, {len(df_existing)} rows")
        return True

    try:
        df = ak.stock_zh_index_daily(symbol=symbol)
        if df is None or df.empty:
            print(f"  [WARN] {name} ({symbol}) returned empty data")
            return False

        # 只保留10年数据 (2015-01-01 ~)
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] >= "2015-01-01"].copy()
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        df = df[["date", "open", "high", "low", "close", "volume"]]
        df.to_csv(out_path, index=False)
        print(f"  [OK]   {name} ({symbol}): {len(df)} rows, "
              f"{df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")
        return True
    except Exception as e:
        print(f"  [ERR]  {name} ({symbol}): {e}")
        return False


def main():
    print(f"=" * 60)
    print(f"下载市场数据 → {OUTPUT_DIR}")
    print(f"品种数: {len(SYMBOLS)}, 每次间隔: {DELAY_SECONDS}s")
    print(f"=" * 60)

    success, fail = 0, 0
    for i, (symbol, name) in enumerate(SYMBOLS):
        print(f"\n[{i+1}/{len(SYMBOLS)}] {name} ({symbol})")
        if download_one(symbol, name):
            success += 1
        else:
            fail += 1

        # 等待20秒 (最后一个不等)
        if i < len(SYMBOLS) - 1:
            print(f"  等待 {DELAY_SECONDS}s...")
            time.sleep(DELAY_SECONDS)

    print(f"\n{'=' * 60}")
    print(f"完成: {success} 成功, {fail} 失败")
    print(f"数据目录: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
