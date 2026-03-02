import sys
import pandas as pd
import optuna

def moving_average(series, n):
    return series.rolling(window=n, min_periods=1).mean()

def crossover(series1, series2):
    return (series1.shift(1) < series2.shift(1)) & (series1 >= series2)

def crossunder(series1, series2):
    return (series1.shift(1) > series2.shift(1)) & (series1 <= series2)

def backtest(df, length0, p1, p2, p3):
    close = df['close']
    openp = df['open']

    avg1 = moving_average(close, length0)
    avg2 = moving_average(close, length0 + p1)
    avg3 = moving_average(close, length0 + p1 + p2)
    avg4 = moving_average(close, length0 + p1 + p2 + p3)

    signals = pd.DataFrame(index=df.index)
    signals['buy'] = (
        crossover(avg1, avg2) | crossover(avg1, avg3) | crossover(avg1, avg4) |
        crossover(avg2, avg3) | crossover(avg2, avg4) | crossover(avg3, avg4)
    )
    signals['sell'] = (
        crossunder(avg1, avg2) | crossunder(avg1, avg3) | crossunder(avg1, avg4) |
        crossunder(avg2, avg3) | crossunder(avg2, avg4) | crossunder(avg3, avg4)
    )

    position = 0
    entry_price = 0
    profit = 0.0

    for i in range(1, len(df)):
        if signals['buy'].iloc[i]:
            if position == -1:
                profit += (entry_price - openp.iloc[i])
                position = 0
            if position == 0:
                entry_price = openp.iloc[i]
                position = 1
        elif signals['sell'].iloc[i]:
            if position == 1:
                profit += (openp.iloc[i] - entry_price)
                position = 0
            if position == 0:
                entry_price = openp.iloc[i]
                position = -1

    if position == 1:
        profit += (close.iloc[-1] - entry_price)
    elif position == -1:
        profit += (entry_price - close.iloc[-1])

    return profit

def main():
    if len(sys.argv) < 2:
        print("Usage: python dual_ma_optuna.py data.csv")
        sys.exit(1)

    filename = sys.argv[1]
    df = pd.read_csv(filename)

    def objective(trial):
        length0 = trial.suggest_int("length0", 3, 15)
        p1 = trial.suggest_int("p1", 2, 15)
        p2 = trial.suggest_int("p2", 2, 15)
        p3 = trial.suggest_int("p3", 2, 15)
        return backtest(df, length0, p1, p2, p3)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=200)  # 迭代 200 次

    print("Best params:", study.best_params)
    print("Best profit:", study.best_value)

if __name__ == "__main__":
    main()

