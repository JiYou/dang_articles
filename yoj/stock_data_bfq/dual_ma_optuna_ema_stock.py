#!/usr/bin/env python3
"""
股票EMA四均线交叉策略回测（只做多） + Optuna参数优化

基于 km_data/dual_ma_optuna_ema.py 策略改编，适配A股不复权数据。
核心区别：
  1. 去掉做空逻辑，仅做多（金叉买入、死叉卖出）
  2. 支持单只股票模式和批量全跑模式
  3. 使用百分比收益率（不复权数据绝对价格差异大）

用法：
  单只股票:  python dual_ma_optuna_ema_stock.py 600900.csv 0.6
  批量全跑:  python dual_ma_optuna_ema_stock.py --batch 0.6 [--n_trials 5000] [--output results.csv]
"""

import sys
import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无GUI后端，适合服务器
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import TPESampler
from datetime import datetime

# 抑制Optuna的日志输出（批量模式下特别重要）
optuna.logging.set_verbosity(optuna.logging.WARNING)


# =============== 工具函数 ===============
def moving_average(series, n):
    """计算指数移动平均线 (EMA)。"""
    if n < 1:
        n = 1
    return series.ewm(span=n, adjust=False, min_periods=1).mean()


def crossover(series1, series2):
    """金叉：前一刻 series1 < series2，当前 series1 >= series2"""
    return (series1.shift(1) < series2.shift(1)) & (series1 >= series2)


def crossunder(series1, series2):
    """死叉：前一刻 series1 > series2，当前 series1 <= series2"""
    return (series1.shift(1) > series2.shift(1)) & (series1 <= series2)


# =============== 策略回测（只做多，修正未来函数） ===============
def strategy_backtest(df, length0, p1, p2, p3, return_extras=False):
    """
    只做多的EMA四均线交叉策略。
    - 任意两条均线金叉 → 买入
    - 任意两条均线死叉 → 卖出
    - 使用百分比收益率，适合不复权数据
    """
    if len(df) < 2:
        return (0.0, None, None, None, None, [], []) if return_extras else 0.0

    close = df['close'].values
    openp = df['open'].values
    dates = df['date'].values

    # 1. 计算四条EMA均线
    close_s = df['close']
    avg1 = moving_average(close_s, length0)
    avg2 = moving_average(close_s, length0 + p1)
    avg3 = moving_average(close_s, length0 + p1 + p2)
    avg4 = moving_average(close_s, length0 + p1 + p2 + p3)

    # 2. 计算原始信号（基于第i天收盘价）
    buy_signal_raw = (
        crossover(avg1, avg2) | crossover(avg1, avg3) | crossover(avg1, avg4) |
        crossover(avg2, avg3) | crossover(avg2, avg4) | crossover(avg3, avg4)
    )
    sell_signal_raw = (
        crossunder(avg1, avg2) | crossunder(avg1, avg3) | crossunder(avg1, avg4) |
        crossunder(avg2, avg3) | crossunder(avg2, avg4) | crossunder(avg3, avg4)
    )

    # 3. 修正未来函数：信号后移一天，次日开盘执行
    buy_signal = buy_signal_raw.shift(1).fillna(False).values
    sell_signal = sell_signal_raw.shift(1).fillna(False).values

    # 4. 回测（只做多）
    position = 0  # 0=空仓, 1=持仓
    entry_price = 0.0
    cum_pct = 0.0  # 累计百分比收益
    trades = []
    pnl_list = [0.0]  # 累计百分比收益序列

    for i in range(1, len(df)):
        daily_pnl = 0.0

        # 持仓期间计算每日浮动百分比盈亏
        if position == 1:
            daily_pnl = (close[i] - close[i-1]) / close[i-1] * 100.0

        # 处理交易信号
        if buy_signal[i] and position == 0:
            # 开多仓
            entry_price = openp[i]
            trades.append((dates[i], "buy", openp[i]))
            position = 1
            # 修正：买入当日的浮动盈亏从开盘价算起
            daily_pnl = (close[i] - openp[i]) / openp[i] * 100.0
        elif sell_signal[i] and position == 1:
            # 平多仓（以开盘价成交）
            realized_pnl = (openp[i] - entry_price) / entry_price * 100.0
            # 当日pnl = 开盘前的浮动（昨收到今开） 
            daily_pnl = (openp[i] - close[i-1]) / close[i-1] * 100.0
            trades.append((dates[i], "sell", openp[i]))
            position = 0

        cum_pct += daily_pnl
        pnl_list.append(cum_pct)

    # 期末平仓
    final_pnl = pnl_list[-1]
    if position == 1:
        final_pnl += (close[-1] - entry_price) / entry_price * 100.0
        trades.append((dates[-1], "sell_end", close[-1]))
        pnl_list[-1] = final_pnl

    if return_extras:
        return final_pnl, avg1, avg2, avg3, avg4, trades, pnl_list
    return final_pnl


# =============== 单只股票模式 ===============
def run_single(filename, split_ratio, n_trials=10000):
    """对单只股票执行Optuna优化 + 可视化"""
    base_filename = os.path.splitext(filename)[0]

    # 1. 加载并分割数据
    df_full = pd.read_csv(filename)
    df_full['date'] = pd.to_datetime(df_full['date'])

    if len(df_full) < 60:
        print(f"⚠️ 数据量不足({len(df_full)}行)，跳过 {filename}")
        return None

    split_index = int(len(df_full) * split_ratio)
    df_train = df_full.iloc[:split_index].copy()
    df_test = df_full.iloc[split_index:].copy().reset_index(drop=True)

    print(f"Total data points: {len(df_full)}")
    print(f"Training: {len(df_train)} ({split_ratio*100:.0f}%) | Testing: {len(df_test)} ({(1-split_ratio)*100:.0f}%)")
    print("-" * 50)

    # 2. Optuna参数优化
    def objective(trial):
        length0 = trial.suggest_int("length0", 3, 15)
        p1 = trial.suggest_int("p1", 2, 15)
        p2 = trial.suggest_int("p2", 2, 15)
        p3 = trial.suggest_int("p3", 2, 15)
        return strategy_backtest(df_train, length0, p1, p2, p3)

    print(f"Starting Optuna optimization ({n_trials} trials)...")
    study = optuna.create_study(direction="maximize", sampler=TPESampler())
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)
    print("Optimization finished.")
    print("-" * 50)

    # 3. 评估结果
    best_params = study.best_params
    in_sample_profit = study.best_value
    out_of_sample_profit = strategy_backtest(df_test, **best_params)

    print("--- In-Sample (Training Set) ---")
    print(f"Best params: {best_params}")
    print(f"Return: {in_sample_profit:.2f}%")
    print("-" * 50)

    print("--- Out-of-Sample (Test Set) ---")
    print(f"Return: {out_of_sample_profit:.2f}%")
    print("-" * 50)

    # 4. 全量数据可视化
    profit_full, avg1, avg2, avg3, avg4, trades, cum_return_full = strategy_backtest(
        df_full, **best_params, return_extras=True
    )

    # 绘制价格+均线图
    plt.figure(figsize=(15, 7))
    plt.plot(df_full['date'], df_full['close'], label="Close", color="black", alpha=0.7)
    plt.plot(df_full['date'], avg1, label=f"EMA{best_params['length0']}")
    plt.plot(df_full['date'], avg2, label=f"EMA{best_params['length0']+best_params['p1']}")
    plt.plot(df_full['date'], avg3, label=f"EMA{best_params['length0']+best_params['p1']+best_params['p2']}")
    plt.plot(df_full['date'], avg4, label=f"EMA{best_params['length0']+best_params['p1']+best_params['p2']+best_params['p3']}")

    split_date = df_full['date'].iloc[split_index]
    plt.axvline(x=split_date, color='r', linestyle='--', label='Train/Test Split')

    # 标注买卖点
    for d, action, price in trades:
        if "buy" in action:
            plt.scatter(d, price, marker="^", color="green", s=60, zorder=5)
        else:
            plt.scatter(d, price, marker="v", color="red", s=60, zorder=5)

    plt.legend()
    plt.title(f"EMA Strategy (Long-Only) | {os.path.basename(filename)}\n"
              f"Params: {best_params} | Total Return: {profit_full:.2f}%")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

    strategy_chart_filename = f"{base_filename}.strategy.png"
    plt.savefig(strategy_chart_filename, dpi=100)
    plt.close()
    print(f"Strategy chart saved: {strategy_chart_filename}")

    # 绘制累计收益曲线
    plt.figure(figsize=(15, 7))
    plt.plot(df_full['date'], cum_return_full, label="Cumulative Return (%)", color="blue")
    plt.axvline(x=split_date, color='r', linestyle='--', label='Train/Test Split')
    plt.axhline(0, ls="--", c="gray")

    plt.title(f"Cumulative Return (%) | {os.path.basename(filename)}\n"
              f"Params: {best_params}")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    pnl_chart_filename = f"{base_filename}.pnl.png"
    plt.savefig(pnl_chart_filename, dpi=100)
    plt.close()
    print(f"PnL chart saved: {pnl_chart_filename}")

    # 尝试保存Optuna图表
    try:
        from optuna.visualization import plot_optimization_history, plot_param_importances
        fig1 = plot_optimization_history(study)
        fig2 = plot_param_importances(study)
        fig1.write_image(f"{base_filename}.optuna_history.png")
        fig2.write_image(f"{base_filename}.optuna_importance.png")
        print(f"Optuna charts saved.")
    except Exception as e:
        print(f"Optuna chart save failed (install kaleido): {e}")

    return {
        'stock': os.path.basename(filename).replace('.csv', ''),
        'best_params': best_params,
        'in_sample_return': in_sample_profit,
        'out_of_sample_return': out_of_sample_profit,
        'full_return': profit_full,
        'n_trades': len(trades),
        'data_points': len(df_full),
    }


# =============== 批量模式 ===============
def run_batch(data_dir, split_ratio, n_trials=5000, output_csv="batch_results.csv"):
    """遍历stock_data_bfq下所有CSV，逐只优化并汇总结果"""
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))

    # 排除本脚本可能产生的结果文件
    csv_files = [f for f in csv_files if not os.path.basename(f).startswith("batch_")]

    print("=" * 60)
    print(f"🚀 批量回测模式")
    print(f"   数据目录: {data_dir}")
    print(f"   股票数量: {len(csv_files)}")
    print(f"   训练比例: {split_ratio}")
    print(f"   每只优化次数: {n_trials}")
    print("=" * 60)

    results = []
    for idx, filepath in enumerate(csv_files, 1):
        stock_code = os.path.basename(filepath).replace('.csv', '')
        print(f"\n[{idx}/{len(csv_files)}] Processing {stock_code}...")

        try:
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])

            if len(df) < 60:
                print(f"  ⚠️ 数据不足({len(df)}行)，跳过")
                continue

            split_index = int(len(df) * split_ratio)
            df_train = df.iloc[:split_index].copy()
            df_test = df.iloc[split_index:].copy().reset_index(drop=True)

            # Optuna优化
            def objective(trial):
                length0 = trial.suggest_int("length0", 3, 15)
                p1 = trial.suggest_int("p1", 2, 15)
                p2 = trial.suggest_int("p2", 2, 15)
                p3 = trial.suggest_int("p3", 2, 15)
                return strategy_backtest(df_train, length0, p1, p2, p3)

            study = optuna.create_study(direction="maximize", sampler=TPESampler())
            study.optimize(objective, n_trials=n_trials, n_jobs=-1, show_progress_bar=False)

            best_params = study.best_params
            in_sample = study.best_value
            out_of_sample = strategy_backtest(df_test, **best_params)
            full_return = strategy_backtest(df, **best_params)

            result = {
                'stock_code': stock_code,
                'data_points': len(df),
                'date_start': df['date'].iloc[0].strftime('%Y-%m-%d'),
                'date_end': df['date'].iloc[-1].strftime('%Y-%m-%d'),
                'length0': best_params['length0'],
                'p1': best_params['p1'],
                'p2': best_params['p2'],
                'p3': best_params['p3'],
                'in_sample_return_pct': round(in_sample, 2),
                'out_of_sample_return_pct': round(out_of_sample, 2),
                'full_return_pct': round(full_return, 2),
            }
            results.append(result)

            print(f"  ✅ IS={in_sample:.1f}% | OOS={out_of_sample:.1f}% | Full={full_return:.1f}% | "
                  f"Params: l0={best_params['length0']}, p1={best_params['p1']}, "
                  f"p2={best_params['p2']}, p3={best_params['p3']}")

        except Exception as e:
            print(f"  ❌ 错误: {e}")
            continue

        # 每100只股票保存一次中间结果
        if idx % 100 == 0:
            df_results = pd.DataFrame(results)
            df_results.to_csv(os.path.join(data_dir, output_csv), index=False)
            print(f"\n  💾 中间结果已保存 ({idx}/{len(csv_files)})")

    # 保存最终结果
    if results:
        df_results = pd.DataFrame(results)
        output_path = os.path.join(data_dir, output_csv)
        df_results.to_csv(output_path, index=False)

        print("\n" + "=" * 60)
        print("🎉 批量回测完成！")
        print(f"   结果已保存: {output_path}")
        print(f"   成功处理: {len(results)}/{len(csv_files)}")
        print("-" * 60)

        # 汇总统计
        print(f"   样本内平均收益: {df_results['in_sample_return_pct'].mean():.2f}%")
        print(f"   样本外平均收益: {df_results['out_of_sample_return_pct'].mean():.2f}%")
        print(f"   样本外>0的股票: {(df_results['out_of_sample_return_pct'] > 0).sum()}/{len(results)} "
              f"({(df_results['out_of_sample_return_pct'] > 0).mean()*100:.1f}%)")
        print(f"   样本外Top10:")
        top10 = df_results.nlargest(10, 'out_of_sample_return_pct')[['stock_code', 'out_of_sample_return_pct', 'in_sample_return_pct']]
        print(top10.to_string(index=False))
        print("=" * 60)
    else:
        print("⚠️ 没有成功处理任何股票。")


# =============== 主程序 ===============
def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    if sys.argv[1] == "--batch":
        # 批量模式
        split_ratio = float(sys.argv[2])
        if not (0 < split_ratio < 1):
            print("Error: split_ratio must be between 0 and 1")
            sys.exit(1)

        n_trials = 5000  # 批量模式默认减少trial数
        output_csv = "batch_results.csv"

        # 解析可选参数
        i = 3
        while i < len(sys.argv):
            if sys.argv[i] == "--n_trials" and i + 1 < len(sys.argv):
                n_trials = int(sys.argv[i + 1])
                i += 2
            elif sys.argv[i] == "--output" and i + 1 < len(sys.argv):
                output_csv = sys.argv[i + 1]
                i += 2
            else:
                print(f"Unknown argument: {sys.argv[i]}")
                sys.exit(1)

        # 数据目录为当前脚本所在目录
        data_dir = os.path.dirname(os.path.abspath(__file__))
        run_batch(data_dir, split_ratio, n_trials, output_csv)
    else:
        # 单只股票模式
        filename = sys.argv[1]
        split_ratio = float(sys.argv[2])
        if not (0 < split_ratio < 1):
            print("Error: split_ratio must be between 0 and 1")
            sys.exit(1)

        n_trials = 10000
        if len(sys.argv) > 3:
            n_trials = int(sys.argv[3])

        run_single(filename, split_ratio, n_trials)


if __name__ == "__main__":
    main()
