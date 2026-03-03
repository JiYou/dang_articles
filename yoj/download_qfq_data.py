"""
用akshare下载前复权数据，用于ETF轮动策略回测。
标的：600030（中信证券）、159941（纳指100ETF）、513100（纳指ETF）
上证指数不需要复权，直接用已有数据。
"""
import akshare as ak
import time
import os

output_dir = "stock_data_qfq"  # 前复权数据目录
os.makedirs(output_dir, exist_ok=True)

# 要下载的标的列表：(symbol, market, filename, description)
targets = [
    ("600030", "sh", "600030.csv", "中信证券"),
    ("159941", "sz", "159941.csv", "纳指100ETF"),
    ("513100", "sh", "513100.csv", "纳指ETF"),
]

for i, (symbol, market, filename, desc) in enumerate(targets):
    if i > 0:
        print(f"等待20秒（akshare限流）...")
        time.sleep(20)

    print(f"下载 {desc} ({symbol})...")
    try:
        # 用 ak.stock_zh_a_hist 获取前复权日线数据
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date="20150101",
            end_date="20260303",
            adjust="qfq"  # 前复权
        )
        
        # 统一列名
        # akshare返回的列: 日期,开盘,收盘,最高,最低,成交量,成交额,振幅,涨跌幅,涨跌额,换手率
        df = df.rename(columns={
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
        })
        
        # 只保留需要的列，按 market_data 格式: date,open,high,low,close,volume
        df = df[["date", "open", "high", "low", "close", "volume"]]
        df = df.sort_values("date").reset_index(drop=True)
        
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"  保存: {filepath}, {len(df)} 行, {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")
        
    except Exception as e:
        print(f"  错误: {e}")
        # 如果stock_zh_a_hist不支持ETF，尝试fund接口
        try:
            print(f"  尝试 fund 接口...")
            time.sleep(5)
            df = ak.fund_etf_hist_em(
                symbol=symbol,
                period="daily",
                start_date="20150101",
                end_date="20260303",
                adjust="qfq"
            )
            df = df.rename(columns={
                "日期": "date",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
            })
            df = df[["date", "open", "high", "low", "close", "volume"]]
            df = df.sort_values("date").reset_index(drop=True)
            
            filepath = os.path.join(output_dir, filename)
            df.to_csv(filepath, index=False)
            print(f"  保存: {filepath}, {len(df)} 行, {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")
        except Exception as e2:
            print(f"  fund接口也失败: {e2}")

print("\n完成！")
