#!/usr/bin/env python3
"""
下载A股所有可交易ETF的日线前复权数据 (via 腾讯财经API)

数据源: web.ifzq.gtimg.cn (腾讯前复权K线接口)
- 每次最多返回800根K线, 通过分页获取全部历史
- 返回前复权 (qfq) 数据, 解决拆分/合并导致的价格跳变问题
- 无需API key, 无严格限流

限流策略:
  - 每只ETF下载完后等待 DELAY_SECONDS (默认3秒)
  - 每个分页请求间等待 PAGE_DELAY (默认0.5秒)
  - 连续失败超过 MAX_CONSECUTIVE_FAILURES 次暂停更长时间

输出: market_data_qfq/{sh|sz}{code}.csv
格式: date,open,high,low,close,volume
"""

import os
import sys
import time
import json
import requests
import pandas as pd
from datetime import datetime


# ====== 配置 ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "market_data_qfq")
PROGRESS_FILE = os.path.join(BASE_DIR, "etf_download_qfq_progress.json")
ETF_LIST_CACHE = os.path.join(BASE_DIR, "etf_list_cache.json")

DELAY_SECONDS = 3              # ETF间等待
PAGE_DELAY = 0.5               # 分页请求间等待
MAX_RETRIES = 3                # 单只ETF最大重试次数
RETRY_BASE_DELAY = 10          # 重试基础延迟(秒)
MAX_CONSECUTIVE_FAILURES = 20  # 连续失败N次后长暂停
LONG_PAUSE_SECONDS = 60        # 长暂停时间
REQUEST_TIMEOUT = 30           # HTTP请求超时(秒)
START_DATE = "2015-01-01"      # 数据起始日期
MAX_PAGES = 15                 # 最大分页次数 (15*800=12000根K线)

TENCENT_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"


def get_exchange_prefix(code: str) -> str:
    """根据ETF代码判断交易所前缀"""
    if code.startswith(("51", "56", "58", "18")):
        return "sh"
    elif code.startswith(("15", "16")):
        return "sz"
    elif code.startswith("0"):
        return "sz"
    elif code.startswith("6"):
        return "sh"
    return "sh"


def fetch_klines_page(symbol: str, end_date: str, count: int = 800) -> list:
    """
    从腾讯API获取一页K线数据 (前复权)

    返回: list of [date, open, close, high, low, volume]
    注意: 腾讯API列顺序为 date,open,close,high,low,volume (非标准OHLCV)
    """
    params = {
        "param": f"{symbol},day,{START_DATE},{end_date},{count},qfq",
        "_var": "kline_dayqfq"
    }
    r = requests.get(TENCENT_URL, params=params, timeout=REQUEST_TIMEOUT)
    text = r.text
    if text.startswith("kline_dayqfq="):
        text = text[len("kline_dayqfq="):]

    data = json.loads(text)

    if not isinstance(data.get("data"), dict):
        return []
    if symbol not in data["data"]:
        return []

    inner = data["data"][symbol]
    if not isinstance(inner, dict):
        return []

    # qfqday = 前复权日线, day = 普通日线 (fallback)
    klines = inner.get("qfqday") or inner.get("day") or []
    return klines


def fetch_all_klines(symbol: str) -> list:
    """
    分页获取某只ETF的全部前复权日线数据

    Returns list of [date, open, close, high, low, volume], sorted by date asc
    """
    all_klines = []
    end_date = datetime.now().strftime("%Y-%m-%d")

    for page in range(MAX_PAGES):
        klines = fetch_klines_page(symbol, end_date)
        if not klines:
            break

        # 去重: 去掉与已有数据重叠的行
        if all_klines:
            earliest_existing = all_klines[0][0]
            klines = [k for k in klines if k[0] < earliest_existing]
            if not klines:
                break

        all_klines = klines + all_klines

        # 不足800行说明已取完全部历史
        if len(klines) < 750:
            break

        # 下一页的结束日期 = 本页最早日期
        end_date = klines[0][0]

        if page < MAX_PAGES - 1:
            time.sleep(PAGE_DELAY)

    return all_klines


def klines_to_dataframe(klines: list) -> pd.DataFrame:
    """
    将腾讯API返回的K线列表转为DataFrame

    腾讯列顺序: [date, open, close, high, low, volume]
    输出列顺序: date, open, high, low, close, volume (标准OHLCV)
    """
    if not klines:
        return pd.DataFrame()

    df = pd.DataFrame(klines, columns=["date", "open", "close", "high", "low", "volume"])

    # 转数值
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 过滤起始日期
    df = df[df["date"] >= START_DATE].copy()

    # 重排为标准OHLCV顺序
    df = df[["date", "open", "high", "low", "close", "volume"]]

    # 按日期排序
    df = df.sort_values("date").reset_index(drop=True)

    return df


def get_etf_list() -> list:
    """从缓存读取ETF列表"""
    if not os.path.exists(ETF_LIST_CACHE):
        print(f"错误: ETF列表缓存不存在: {ETF_LIST_CACHE}")
        print("请先运行 download_all_etfs.py 生成缓存, 或手动创建")
        sys.exit(1)

    with open(ETF_LIST_CACHE, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"  从缓存加载 {len(records)} 只ETF ({ETF_LIST_CACHE})")
    return records


def download_one_etf(code: str, name: str) -> bool:
    """下载单只ETF的前复权数据, 带重试"""
    prefix = get_exchange_prefix(code)
    symbol = f"{prefix}{code}"
    filename = f"{symbol}.csv"
    out_path = os.path.join(OUTPUT_DIR, filename)

    # 跳过已存在且有足够数据的文件
    if os.path.exists(out_path):
        try:
            df_existing = pd.read_csv(out_path)
            if len(df_existing) >= 10:
                print(f"  [SKIP] {name} ({symbol}) 已存在, {len(df_existing)} 行")
                return True
        except Exception:
            os.remove(out_path)
            print(f"  [WARN] {name} ({symbol}) 文件损坏, 重新下载")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            klines = fetch_all_klines(symbol)
            df = klines_to_dataframe(klines)

            if df.empty:
                print(f"  [WARN] {name} ({symbol}) 无数据")
                return False

            # 过滤异常数据 (价格<=0)
            df = df[(df["close"] > 0) & (df["open"] > 0)].copy()

            if df.empty:
                print(f"  [WARN] {name} ({symbol}) 过滤后无有效数据")
                return False

            df.to_csv(out_path, index=False)
            print(f"  [OK]   {name} ({symbol}): {len(df)} 行, "
                  f"{df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")
            return True

        except requests.exceptions.Timeout:
            print(f"  [ERR]  {name} ({symbol}) 尝试{attempt}/{MAX_RETRIES}: 请求超时")
        except requests.exceptions.ConnectionError as e:
            print(f"  [ERR]  {name} ({symbol}) 尝试{attempt}/{MAX_RETRIES}: 连接错误 {e}")
        except json.JSONDecodeError as e:
            print(f"  [ERR]  {name} ({symbol}) 尝试{attempt}/{MAX_RETRIES}: JSON解析错误 {e}")
        except Exception as e:
            print(f"  [ERR]  {name} ({symbol}) 尝试{attempt}/{MAX_RETRIES}: {type(e).__name__}: {e}")

        if attempt < MAX_RETRIES:
            wait = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            print(f"         等待 {wait}s 后重试...")
            time.sleep(wait)

    print(f"  [FAIL] {name} ({symbol}) 全部 {MAX_RETRIES} 次尝试失败")
    return False


def save_progress(completed: list, failed: list, skipped: int, total: int):
    """保存进度到文件"""
    progress = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total": total,
        "completed_count": len(completed),
        "skipped_count": skipped,
        "failed_count": len(failed),
        "completed": completed,
        "failed": failed,
    }
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70, flush=True)
    print("下载A股全部ETF日线前复权数据 (腾讯财经源)", flush=True)
    print(f"输出目录: {OUTPUT_DIR}", flush=True)
    print(f"请求间隔: {DELAY_SECONDS}s, 分页间隔: {PAGE_DELAY}s", flush=True)
    print(f"重试次数: {MAX_RETRIES}, 超时: {REQUEST_TIMEOUT}s", flush=True)
    print(f"数据范围: {START_DATE} ~ 今", flush=True)
    print("=" * 70, flush=True)

    # Step 1: 获取ETF列表
    print("\n[Step 1] 获取全部ETF列表...", flush=True)
    records = get_etf_list()

    etf_codes = [r["code"] for r in records]
    etf_names = [r["name"] for r in records]

    total = len(etf_codes)
    est_minutes = total * DELAY_SECONDS / 60
    print(f"\n共 {total} 只ETF待下载", flush=True)
    print(f"预计耗时: {est_minutes:.0f} 分钟 (不含重试和跳过)", flush=True)
    print("-" * 70, flush=True)

    # Step 2: 逐一下载
    completed = []
    failed = []
    skipped = 0
    consecutive_failures = 0
    download_count = 0

    for i, (code, name) in enumerate(zip(etf_codes, etf_names)):
        print(f"\n[{i+1}/{total}] {name} ({code})", flush=True)

        # 快速跳过已存在文件
        prefix = get_exchange_prefix(code)
        out_path = os.path.join(OUTPUT_DIR, f"{prefix}{code}.csv")
        if os.path.exists(out_path):
            try:
                df_check = pd.read_csv(out_path)
                if len(df_check) >= 10:
                    print(f"  [SKIP] {name} ({prefix}{code}) 已存在, {len(df_check)} 行", flush=True)
                    skipped += 1
                    completed.append(f"{code}:{name}")
                    consecutive_failures = 0
                    continue
            except Exception:
                pass

        # 实际下载 - 第二只开始才等待
        if download_count > 0:
            print(f"  等待 {DELAY_SECONDS}s...", flush=True)
            time.sleep(DELAY_SECONDS)

        success = download_one_etf(code, name)
        download_count += 1

        if success:
            completed.append(f"{code}:{name}")
            consecutive_failures = 0
        else:
            failed.append(f"{code}:{name}")
            consecutive_failures += 1

            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"\n{'!'*50}", flush=True)
                print(f"  连续失败 {consecutive_failures} 次, 暂停 {LONG_PAUSE_SECONDS}s", flush=True)
                print(f"{'!'*50}", flush=True)
                time.sleep(LONG_PAUSE_SECONDS)
                consecutive_failures = 0

        # 每100只保存一次进度
        if (i + 1) % 100 == 0:
            save_progress(completed, failed, skipped, total)
            print(f"\n  [进度] {i+1}/{total} | 成功: {len(completed)} | "
                  f"跳过: {skipped} | 失败: {len(failed)}", flush=True)

    # Step 3: 保存最终进度
    save_progress(completed, failed, skipped, total)

    # 汇总
    print(f"\n{'=' * 70}", flush=True)
    print(f"下载完成!", flush=True)
    print(f"  总数: {total}", flush=True)
    print(f"  成功(含跳过): {len(completed)}", flush=True)
    print(f"  其中跳过: {skipped}", flush=True)
    print(f"  新下载: {len(completed) - skipped}", flush=True)
    print(f"  失败: {len(failed)}", flush=True)
    print(f"  数据目录: {OUTPUT_DIR}", flush=True)
    if failed:
        print(f"\n失败列表:", flush=True)
        for f_item in failed[:50]:
            print(f"  - {f_item}", flush=True)
        if len(failed) > 50:
            print(f"  ... 及其他 {len(failed)-50} 只", flush=True)
    print(f"{'=' * 70}", flush=True)


if __name__ == "__main__":
    main()
