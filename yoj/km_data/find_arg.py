import sys
import pandas as pd
import numpy as np
from itertools import product

# 计算均线
def moving_average(series, n):
    return series.rolling(window=n, min_periods=1).mean()

# 检测上穿
def crossover(series1, series2):
    return (series1.shift(1) < series2.shift(1)) & (series1 >= series2)

# 检测下穿
def crossunder(series1, series2):
    return (series1.shift(1) > series2.shift(1)) & (series1 <= series2)

# 策略回测函数
def backtest(df, length0, p1, p2, p3):
    close = df['close']
    openp = df['open']

    # 四条均线
    avg1 = moving_average(close, length0)
    avg2 = moving_average(close, length0 + p1)
    avg3 = moving_average(close, length0 + p1 + p2)
    avg4 = moving_average(close, length0 + p1 + p2 + p3)

    # 交叉信号
    signals = pd.DataFrame(index=df.index)
    signals['buy'] = (
        crossover(avg1, avg2) | crossover(avg1, avg3) | crossover(avg1, avg4) |
        crossover(avg2, avg3) | crossover(avg2, avg4) | crossover(avg3, avg4)
    )
    signals['sell'] = (
        crossunder(avg1, avg2) | crossunder(avg1, avg3) | crossunder(avg1, avg4) |
        crossunder(avg2, avg3) | crossunder(avg2, avg4) | crossunder(avg3, avg4)
    )

    position = 0   # 1 = 多头, -1 = 空头, 0 = 空仓
    entry_price = 0
    profit = 0.0

    for i in range(1, len(df)):
        if signals['buy'].iloc[i]:
            if position == -1:  # 平空
                profit += (entry_price - openp.iloc[i])
                position = 0
            if position == 0:   # 开多
                entry_price = openp.iloc[i]
                position = 1
        elif signals['sell'].iloc[i]:
            if position == 1:  # 平多
                profit += (openp.iloc[i] - entry_price)
                position = 0
            if position == 0:  # 开空
                entry_price = openp.iloc[i]
                position = -1

    # 最后平仓
    if position == 1:
        profit += (close.iloc[-1] - entry_price)
    elif position == -1:
        profit += (entry_price - close.iloc[-1])

    return profit

def main():
    if len(sys.argv) < 2:
        print("Usage: python dual_ma_optimize.py data.csv")
        sys.exit(1)

    filename = sys.argv[1]
    df = pd.read_csv(filename)
    df = df[['date','time','open','high','low','close']]

    best_params = None
    best_profit = -1e18

    # 参数搜索范围（你可以根据需要调整）
    for length0, p1, p2, p3 in product(range(3, 15), range(2, 15), range(2, 15), range(2, 15)):
        profit = backtest(df, length0, p1, p2, p3)
        if profit > best_profit:
            best_profit = profit
            best_params = (length0, p1, p2, p3)

    print("最佳参数:", best_params, "收益:", best_profit)

if __name__ == "__main__":
    main()

