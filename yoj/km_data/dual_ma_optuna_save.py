import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import TPESampler
import plotly.io as pio
from optuna.visualization import plot_optimization_history, plot_param_importances

# 注意：为了让 Plotly 能够将图表保存为图片，您需要在服务器上安装 'kaleido' 包。
# pip install kaleido

# =============== 工具函数 ===============
def moving_average(series, n):
    return series.rolling(window=n, min_periods=1).mean()

def crossover(series1, series2):
    return (series1.shift(1) < series2.shift(1)) & (series1 >= series2)

def crossunder(series1, series2):
    return (series1.shift(1) > series2.shift(1)) & (series1 <= series2)

# =============== 策略回测（用于优化和最终评估） ===============
def strategy_backtest(df, length0, p1, p2, p3, return_extras=False):
    if len(df) < 2:
        return (0, None, None, None, None, [], []) if return_extras else 0

    close = df['close']
    openp = df['open']

    avg1 = moving_average(close, length0)
    avg2 = moving_average(close, length0 + p1)
    avg3 = moving_average(close, length0 + p1 + p2)
    avg4 = moving_average(close, length0 + p1 + p2 + p3)

    # 2. 计算原始信号
    # 此时，buy_signal_raw.iloc[i] 是基于第i天收盘价才能确定的信号
    buy_signal_raw = (
        crossover(avg1, avg2) | crossover(avg1, avg3) | crossover(avg1, avg4) |
        crossover(avg2, avg3) | crossover(avg2, avg4) | crossover(avg3, avg4)
    )
    sell_signal_raw = (
        crossunder(avg1, avg2) | crossunder(avg1, avg3) | crossunder(avg1, avg4) |
        crossunder(avg2, avg3) | crossunder(avg2, avg4) | crossunder(avg3, avg4)
    )

    # ==================== 修正未来函数的关键步骤 ====================
    # 3. 将信号整体向后移动一个周期。
    #    这样，在第 i 天，我们使用的是基于 i-1 天收盘数据生成的信号。
    #    这确保了我们在第 i 天开盘交易时，没有使用未来的信息。
    #    .fillna(False) 用于处理第一行移动后产生的NaN值。
    buy_signal = buy_signal_raw.shift(1).fillna(False)
    sell_signal = sell_signal_raw.shift(1).fillna(False)
    # ==================== 修正结束 ==================================

    position = 0
    entry_price = 0
    profit = 0.0
    trades = []  # 记录交易点 (日期, 动作, 价格)
    pnl_list = [0.0] # 记录每日累计盈亏

    for i in range(1, len(df)):
        daily_pnl = 0.0
        # 1. 计算持仓浮动盈亏
        if position == 1:
            daily_pnl = close.iloc[i] - close.iloc[i-1]
        elif position == -1:
            daily_pnl = close.iloc[i-1] - close.iloc[i]

        # 2. 处理交易信号（开盘价成交）
        if buy_signal.iloc[i]:
            if position == -1:  # 平空
                realized_pnl = entry_price - openp.iloc[i]
                daily_pnl += realized_pnl
                trades.append((df['date'].iloc[i], "cover", openp.iloc[i]))
                position = 0
            if position == 0:   # 开多
                entry_price = openp.iloc[i]
                trades.append((df['date'].iloc[i], "buy", openp.iloc[i]))
                position = 1
        elif sell_signal.iloc[i]:
            if position == 1:   # 平多
                realized_pnl = openp.iloc[i] - entry_price
                daily_pnl += realized_pnl
                trades.append((df['date'].iloc[i], "sell", openp.iloc[i]))
                position = 0
            if position == 0:   # 开空
                entry_price = openp.iloc[i]
                trades.append((df['date'].iloc[i], "short", openp.iloc[i]))
                position = -1
        
        pnl_list.append(pnl_list[-1] + daily_pnl)

    # 在数据段结束时平仓
    final_pnl = pnl_list[-1]
    if position == 1:
        final_pnl += (close.iloc[-1] - entry_price)
        trades.append((df['date'].iloc[-1], "sell_end", close.iloc[-1]))
    elif position == -1:
        final_pnl += (entry_price - close.iloc[-1])
        trades.append((df['date'].iloc[-1], "cover_end", close.iloc[-1]))
    
    # 如果有期末平仓，更新最后一个pnl值
    if len(pnl_list) > 0:
        pnl_list[-1] = final_pnl

    if return_extras:
        return final_pnl, avg1, avg2, avg3, avg4, trades, pnl_list
    return final_pnl


# =============== 主程序 ===============
def main():
    if len(sys.argv) < 3:
        print("Usage: python your_script_name.py data.csv split_ratio")
        print("Example: python your_script_name.py a9888.DCE1d.txt 0.6")
        sys.exit(1)

    filename = sys.argv[1]
    # 从输入文件名中提取基础名称，用于生成输出文件名
    base_filename = os.path.splitext(filename)[0]

    try:
        split_ratio = float(sys.argv[2])
        if not (0 < split_ratio < 1):
            raise ValueError("Split ratio must be between 0 and 1.")
    except ValueError as e:
        print(f"Error: Invalid split_ratio. {e}")
        sys.exit(1)

    # 1. 加载并分割数据
    df_full = pd.read_csv(filename)
    df_full['date'] = pd.to_datetime(df_full['date'])

    split_index = int(len(df_full) * split_ratio)
    df_train = df_full.iloc[:split_index].copy()
    df_test = df_full.iloc[split_index:].copy().reset_index(drop=True)

    print(f"Total data points: {len(df_full)}")
    print(f"Training data points: {len(df_train)} (first {split_ratio*100:.0f}%)")
    print(f"Testing data points: {len(df_test)} (last {(1-split_ratio)*100:.0f}%)")
    print("-" * 30)

    # 2. 在训练集上进行参数优化
    # ==================== 修正开始：在这里重新加入 objective 函数的定义 ====================
    def objective(trial):
        length0 = trial.suggest_int("length0", 3, 15)
        p1 = trial.suggest_int("p1", 2, 15)
        p2 = trial.suggest_int("p2", 2, 15)
        p3 = trial.suggest_int("p3", 2, 15)
        # 优化过程只使用训练集 (df_train)
        return strategy_backtest(df_train, length0, p1, p2, p3)
    # ==================== 修正结束 =================================================

    print("Starting Optuna optimization on the training set...")
    study = optuna.create_study(direction="maximize", sampler=TPESampler())
    study.optimize(objective, n_trials=10000, n_jobs=-1) # 使用所有CPU核心并行计算
    print("Optimization finished.")
    print("-" * 30)

    # 3. 评估最优参数在训练集和测试集上的表现
    best_params = study.best_params
    in_sample_profit = study.best_value
    out_of_sample_profit = strategy_backtest(df_test, **best_params)

    print("--- In-Sample Results (Training Set) ---")
    print(f"Optimal parameters found: {best_params}")
    print(f"Profit on training set: {in_sample_profit:.2f}")
    print("-" * 30)
    
    print("--- Out-of-Sample Results (Test Set) ---")
    print(f"Applying parameters {best_params} to the test set.")
    print(f"Profit on test set: {out_of_sample_profit:.2f}")
    print("-" * 30)

    # 4. 在完整数据集上可视化最优策略
    profit_full, avg1, avg2, avg3, avg4, trades, cum_return_full = strategy_backtest(
        df_full, **best_params, return_extras=True
    )

    # 绘制价格、均线和交易信号图
    plt.figure(figsize=(15, 7))
    plt.plot(df_full['date'], df_full['close'], label="Close", color="black", alpha=0.7)
    plt.plot(df_full['date'], avg1, label=f"MA{best_params['length0']}")
    plt.plot(df_full['date'], avg2, label=f"MA{best_params['length0']+best_params['p1']}")
    plt.plot(df_full['date'], avg3, label=f"MA{best_params['length0']+best_params['p1']+best_params['p2']}")
    plt.plot(df_full['date'], avg4, label=f"MA{best_params['length0']+best_params['p1']+best_params['p2']+best_params['p3']}")
    
    # 绘制训练/测试分割线
    split_date = df_full['date'].iloc[split_index]
    plt.axvline(x=split_date, color='r', linestyle='--', label=f'Train/Test Split')

    plt.legend()
    plt.title(f"Strategy on Full Data with Best Params {best_params}\nTotal Profit={profit_full:.2f}")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    
    # 保存图表而不是显示
    strategy_chart_filename = f"{base_filename}.strategy.png"
    plt.savefig(strategy_chart_filename)
    plt.close() # 关闭图形，释放内存
    print(f"Strategy chart saved to: {strategy_chart_filename}")

    # 5. 绘制累计收益曲线
    plt.figure(figsize=(15, 7))
    plt.plot(df_full['date'], cum_return_full, label="Cumulative PnL")
    plt.axvline(x=split_date, color='r', linestyle='--', label=f'Train/Test Split')
    plt.axhline(0, ls="--", c="gray")
    
    plt.title(f"Cumulative PnL with Best Params {best_params}")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Profit/Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 保存图表而不是显示
    pnl_chart_filename = f"{base_filename}.pnl.png"
    plt.savefig(pnl_chart_filename)
    plt.close() # 关闭图形，释放内存
    print(f"Cumulative PnL chart saved to: {pnl_chart_filename}")

    # 6. Plotly 可视化优化过程并保存为文件
    print("Generating Optuna visualization files...")
    fig1 = plot_optimization_history(study)
    fig2 = plot_param_importances(study)

    # 定义输出文件名
    optuna_history_filename = f"{base_filename}.optuna_history.png"
    optuna_importance_filename = f"{base_filename}.optuna_importance.png"

    # 保存图表而不是显示
    try:
        fig1.write_image(optuna_history_filename)
        print(f"Optuna optimization history saved to: {optuna_history_filename}")
        fig2.write_image(optuna_importance_filename)
        print(f"Optuna parameter importances saved to: {optuna_importance_filename}")
    except Exception as e:
        print("\n---")
        print(f"Error saving Plotly images: {e}")
        print("Please make sure you have the 'kaleido' package installed (`pip install kaleido`).")
        print("---\n")


if __name__ == "__main__":
    main()

