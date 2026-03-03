#!/usr/bin/env python3
"""
下载A股所有可交易ETF的日线数据 (via akshare Sina源)

流程:
  1. 通过 fund_etf_category_sina("ETF基金") 获取全部ETF列表
  2. 逐一通过 fund_etf_hist_sina(symbol) 下载日线历史
  3. 每只ETF下载完后等待20秒, 严格串行

注意:
  - 使用Sina源 (东方财富源被限流)
  - fund_etf_hist_sina 返回不复权数据 (绝大多数ETF无拆合并,不影响)
  - 少数有拆分的ETF (如513100, 159941) 后续需单独处理前复权
  - 使用multiprocessing做超时控制 (signal.alarm对阻塞I/O无效)

限流策略:
  - 每次请求间隔 DELAY_SECONDS (默认20秒)
  - 失败时指数退避重试, 最多 MAX_RETRIES 次
  - 连续失败超过 MAX_CONSECUTIVE_FAILURES 次则暂停更长时间

输出: market_data/{sh|sz}{code}.csv
格式: date,open,high,low,close,volume
"""

import os
import sys
import time
import json
import multiprocessing as mp
from datetime import datetime

import pandas as pd


# ====== 配置 ======
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "market_data")
PROGRESS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "etf_download_progress.json")
ETF_LIST_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "etf_list_cache.json")

DELAY_SECONDS = 20          # 每次请求间隔
MAX_RETRIES = 5             # 单只ETF最大重试次数
RETRY_BASE_DELAY = 30       # 重试基础延迟(秒), 指数递增
MAX_CONSECUTIVE_FAILURES = 10  # 连续失败N次后长暂停
LONG_PAUSE_SECONDS = 300    # 长暂停时间(5分钟)
API_TIMEOUT = 90            # 单次API调用超时(秒)


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


# ====== 子进程 worker 函数 (在独立进程中运行, 可被kill) ======

def _fetch_etf_list(result_queue):
    """子进程: 获取ETF列表"""
    try:
        import akshare as ak
        df = ak.fund_etf_category_sina(symbol="ETF基金")
        if df is not None and not df.empty:
            # 序列化为dict传回
            result_queue.put(("ok", df.to_dict()))
        else:
            result_queue.put(("empty", None))
    except Exception as e:
        result_queue.put(("error", str(e)))


def _fetch_etf_hist(symbol, result_queue):
    """子进程: 下载单只ETF历史数据"""
    try:
        import akshare as ak
        df = ak.fund_etf_hist_sina(symbol=symbol)
        if df is not None and not df.empty:
            result_queue.put(("ok", df.to_dict()))
        else:
            result_queue.put(("empty", None))
    except Exception as e:
        result_queue.put(("error", str(e)))


def fetch_with_timeout(target_func, args, timeout=API_TIMEOUT):
    """在子进程中执行API调用, 超时可靠kill

    返回: (status, data) 其中 status 为 'ok'/'empty'/'error'/'timeout'
    """
    result_queue = mp.Queue()
    proc = mp.Process(target=target_func, args=(*args, result_queue))
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.kill()
        proc.join(timeout=5)
        return ("timeout", f"API call timed out after {timeout}s")

    try:
        if not result_queue.empty():
            return result_queue.get_nowait()
        else:
            return ("error", "No result returned from subprocess")
    except Exception as e:
        return ("error", f"Queue error: {e}")


def get_etf_list() -> list:
    """获取全部ETF列表: 优先从缓存读取, 否则从API获取"""
    # 优先读缓存
    if os.path.exists(ETF_LIST_CACHE):
        with open(ETF_LIST_CACHE, 'r', encoding='utf-8') as f:
            records = json.load(f)
        print(f"  从缓存加载 {len(records)} 只ETF ({ETF_LIST_CACHE})")
        return records

    # 缓存不存在, 从API获取
    for attempt in range(1, MAX_RETRIES + 1):
        print(f"  获取ETF列表... (尝试 {attempt}/{MAX_RETRIES})")
        status, data = fetch_with_timeout(_fetch_etf_list, ())

        if status == "ok":
            df = pd.DataFrame(data)
            records = []
            for _, row in df.iterrows():
                code = str(row['代码']).replace('sh','').replace('sz','')
                records.append({'code': code.zfill(6), 'name': str(row['名称'])})
            # 保存缓存
            with open(ETF_LIST_CACHE, 'w', encoding='utf-8') as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
            print(f"  成功获取 {len(records)} 只ETF, 已缓存")
            return records
        elif status == "empty":
            print(f"  返回空数据")
        elif status == "timeout":
            print(f"  超时: {data}")
        else:
            print(f"  失败: {data}")

        if attempt < MAX_RETRIES:
            wait = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            print(f"  等待 {wait}s 后重试...")
            time.sleep(wait)

    print("  无法获取ETF列表, 退出")
    sys.exit(1)


def download_one_etf(code: str, name: str) -> bool:
    """下载单只ETF, 带重试和指数退避"""
    prefix = get_exchange_prefix(code)
    symbol = f"{prefix}{code}"
    filename = f"{symbol}.csv"
    out_path = os.path.join(OUTPUT_DIR, filename)

    # 跳过已存在且非空的文件
    if os.path.exists(out_path):
        try:
            df_existing = pd.read_csv(out_path)
            if len(df_existing) > 0:
                print(f"  [SKIP] {name} ({symbol}) 已存在, {len(df_existing)} 行")
                return True
        except Exception:
            os.remove(out_path)
            print(f"  [WARN] {name} ({symbol}) 文件损坏, 重新下载")

    for attempt in range(1, MAX_RETRIES + 1):
        status, data = fetch_with_timeout(_fetch_etf_hist, (symbol,))

        if status == "ok":
            df = pd.DataFrame(data)
            # fund_etf_hist_sina 返回: date, open, high, low, close, volume
            df = df[["date", "open", "high", "low", "close", "volume"]]

            # 过滤2015年以后
            df["date"] = pd.to_datetime(df["date"])
            df = df[df["date"] >= "2015-01-01"].copy()
            df["date"] = df["date"].dt.strftime("%Y-%m-%d")

            if df.empty:
                print(f"  [WARN] {name} ({symbol}) 2015年后无数据")
                return False

            df.to_csv(out_path, index=False)
            print(f"  [OK]   {name} ({symbol}): {len(df)} 行, "
                  f"{df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")
            return True

        elif status == "empty":
            print(f"  [WARN] {name} ({symbol}) 返回空数据")
            return False

        else:
            print(f"  [ERR]  {name} ({symbol}) 尝试{attempt}/{MAX_RETRIES}: {data}")
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
    print("下载A股全部ETF日线数据 (Sina源)", flush=True)
    print(f"输出目录: {OUTPUT_DIR}", flush=True)
    print(f"请求间隔: {DELAY_SECONDS}s, 重试次数: {MAX_RETRIES}, 超时: {API_TIMEOUT}s", flush=True)
    print(f"数据范围: 2015-01-01 ~ 今", flush=True)
    print("=" * 70, flush=True)

    # Step 1: 获取ETF列表
    print("\n[Step 1] 获取全部ETF列表...", flush=True)
    records = get_etf_list()

    etf_codes = [r['code'] for r in records]
    etf_names = [r['name'] for r in records]

    total = len(etf_codes)
    print(f"\n共 {total} 只ETF待下载", flush=True)
    est_hours = total * DELAY_SECONDS / 3600
    print(f"预计耗时: {est_hours:.1f} 小时 (不含重试和跳过)", flush=True)
    print("-" * 70, flush=True)

    # 短暂等待
    print(f"\n等待 {DELAY_SECONDS}s 后开始下载...", flush=True)
    time.sleep(DELAY_SECONDS)

    # Step 2: 逐一下载
    completed = []
    failed = []
    skipped = 0
    consecutive_failures = 0
    download_count = 0  # 实际发起下载请求的计数

    for i, (code, name) in enumerate(zip(etf_codes, etf_names)):
        print(f"\n[{i+1}/{total}] {name} ({code})", flush=True)

        # 快速跳过已存在文件 (不计延迟)
        prefix = get_exchange_prefix(code)
        out_path = os.path.join(OUTPUT_DIR, f"{prefix}{code}.csv")
        if os.path.exists(out_path):
            try:
                df_check = pd.read_csv(out_path)
                if len(df_check) > 0:
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

        # 每50只保存一次进度
        if (i + 1) % 50 == 0:
            save_progress(completed, failed, skipped, total)
            print(f"\n  [进度] {i+1}/{total} | 成功: {len(completed)} | 跳过: {skipped} | 失败: {len(failed)}", flush=True)

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
        for f_item in failed:
            print(f"  - {f_item}", flush=True)
    print(f"{'=' * 70}", flush=True)


if __name__ == "__main__":
    mp.set_start_method("fork")
    main()
