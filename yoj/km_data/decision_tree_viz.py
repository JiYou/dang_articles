import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# =============== 工具函数 ===============
def moving_average(series, n):
    """计算移动平均线"""
    return series.rolling(window=n, min_periods=1).mean()

# =============== ML策略所需的新函数 ===============

def create_features(df, length0, p1, p2, p3):
    """
    根据原始数据和均线参数创建特征集(Features)
    这是将均线数据转化为机器学习模型输入的核心步骤。
    """
    features = pd.DataFrame(index=df.index)
    close = df['close']

    # 1. 计算基础均线
    avg1 = moving_average(close, length0)
    avg2 = moving_average(close, length0 + p1)
    avg3 = moving_average(close, length0 + p1 + p2)
    avg4 = moving_average(close, length0 + p1 + p2 + p3)

    # 2. 创建特征 (这里只是一些例子，可以无限扩展)
    # 特征1: 价格与均线的相对位置 (乖离率思想)
    features['price_ma1_ratio'] = close / avg1
    features['price_ma4_ratio'] = close / avg4
    
    # 特征2: 不同周期均线间的相对关系 (趋势结构)
    features['ma1_ma2_ratio'] = avg1 / avg2
    features['ma1_ma4_ratio'] = avg1 / avg4
    features['ma2_ma4_ratio'] = avg2 / avg4
    
    # 特征3: 均线的斜率/动量 (趋势强度)
    features['ma1_slope'] = avg1.diff(5) # 5日变化量
    features['ma4_slope'] = avg4.diff(5)

    # 删除因计算(如diff, rolling)产生的NaN值，确保数据干净
    features = features.dropna()
    
    return features

def create_labels(df, look_forward_period=10, threshold=0.05):
    """
    为数据创建标签(Labels)，即模型的学习目标。
    - look_forward_period: 预测未来多少天的走势
    - threshold: 决定买入/卖出的收益率阈值
    
    标签定义:
    - 1 (买入): 如果未来N天价格上涨超过阈值
    - -1 (卖出): 如果未来N天价格下跌超过阈值
    - 0 (持有): 其他情况
    """
    # 计算未来N天的收益率
    future_returns = df['close'].shift(-look_forward_period) / df['close'] - 1
    
    labels = pd.Series(0, index=df.index)  # 默认标签为 0 (持有)
    labels.loc[future_returns > threshold] = 1   # 标记为 1 (买入)
    labels.loc[future_returns < -threshold] = -1 # 标记为 -1 (卖出)
    
    return labels

def strategy_backtest_ml(df, signals):
    """
    根据机器学习模型预测的信号进行回测。
    """
    if len(df) < 2:
        return 0, [], []

    position = 0
    entry_price = 0
    profit = 0.0
    trades = []  # 记录交易点 (日期, 动作, 价格)
    pnl_list = [0.0] # 记录每日累计盈亏

    # 将预测信号合并到主DataFrame中，并用0填充缺失值
    df = df.copy()
    df['signal'] = signals
    df['signal'] = df['signal'].fillna(0)

    for i in range(1, len(df)):
        daily_pnl = 0.0
        # 1. 计算持仓浮动盈亏
        if position == 1:
            daily_pnl = df['close'].iloc[i] - df['close'].iloc[i-1]
        elif position == -1:
            daily_pnl = df['close'].iloc[i-1] - df['close'].iloc[i]

        # 2. 处理交易信号（基于前一天的信号，在当天开盘价成交）
        current_signal = df['signal'].iloc[i-1]
        
        if current_signal == 1: # 买入信号
            if position == -1:  # 如果有空仓，先平仓
                realized_pnl = entry_price - df['open'].iloc[i]
                daily_pnl += realized_pnl
                trades.append((df['date'].iloc[i], "cover", df['open'].iloc[i]))
                position = 0
            if position == 0:   # 开多仓
                position = 1
                entry_price = df['open'].iloc[i]
                trades.append((df['date'].iloc[i], "buy", entry_price))
        
        elif current_signal == -1: # 卖出信号
            if position == 1:   # 如果有多仓，先平仓
                realized_pnl = df['open'].iloc[i] - entry_price
                daily_pnl += realized_pnl
                trades.append((df['date'].iloc[i], "sell", df['open'].iloc[i]))
                position = 0
            if position == 0:   # 开空仓
                position = -1
                entry_price = df['open'].iloc[i]
                trades.append((df['date'].iloc[i], "short", entry_price))
        
        pnl_list.append(pnl_list[-1] + daily_pnl)

    # 在数据段结束时强制平仓
    final_pnl = pnl_list[-1]
    if position == 1:
        final_pnl += (df['close'].iloc[-1] - entry_price)
        trades.append((df['date'].iloc[-1], "sell_end", df['close'].iloc[-1]))
    elif position == -1:
        final_pnl += (entry_price - df['close'].iloc[-1])
        trades.append((df['date'].iloc[-1], "cover_end", df['close'].iloc[-1]))
    
    if len(pnl_list) > 0:
        pnl_list[-1] = final_pnl

    return final_pnl, trades, pnl_list


# =============== 主程序 ===============
def main():
    if len(sys.argv) < 3:
        print("Usage: python your_script_name.py data.csv split_ratio")
        print("Example: python your_script_name.py a9888.DCE1d.txt 0.6")
        sys.exit(1)

    filename = sys.argv[1]
    split_ratio = float(sys.argv[2])

    # 1. 加载并分割数据
    df_full = pd.read_csv(filename)
    df_full['date'] = pd.to_datetime(df_full['date'])

    split_index = int(len(df_full) * split_ratio)
    df_train = df_full.iloc[:split_index].copy()
    df_test = df_full.iloc[split_index:].copy().reset_index(drop=True)

    print(f"Total data points: {len(df_full)}")
    print(f"Training data points: {len(df_train)}")
    print(f"Testing data points: {len(df_test)}")
    print("-" * 30)

    # 2. 定义策略超参数
    # 这些参数可以像你之前一样使用Optuna进行优化
    ma_params = {'length0': 10, 'p1': 5, 'p2': 10, 'p3': 15}
    label_params = {'look_forward_period': 10, 'threshold': 0.05}
    
    print("--- Step 1: Preparing Data and Training Model on Training Set ---")
    
    # 3. 在训练集上创建特征和标签
    features_train = create_features(df_train, **ma_params)
    labels_train = create_labels(df_train, **label_params)
    
    # 对齐特征和标签，确保每个特征向量都有一个对应的标签
    common_index = features_train.index.intersection(labels_train.index)
    X_train = features_train.loc[common_index]
    y_train = labels_train.loc[common_index]

    print(f"Number of training samples: {len(X_train)}")
    
    # 4. 训练决策树模型
    # max_depth是一个重要参数，用于防止过拟合，可以进行调优
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    print("\nModel Training Report (on training data):")
    train_pred = model.predict(X_train)
    print(classification_report(y_train, train_pred, target_names=['Sell (-1)', 'Hold (0)', 'Buy (1)']))
    print("-" * 30)

    # 5. 在测试集上进行预测和回测
    print("--- Step 2: Backtesting on Test Set using the Trained Model ---")
    
    # 为测试集创建同样的特征
    features_test = create_features(df_test, **ma_params)
    X_test = features_test.loc[features_test.index]

    # 使用训练好的模型进行预测
    predicted_signals = pd.Series(model.predict(X_test), index=X_test.index)
    
    # 使用ML回测函数进行评估
    test_profit, test_trades, test_pnl = strategy_backtest_ml(df_test, predicted_signals)
    
    print(f"Out-of-Sample Profit (Test Set): {test_profit:.2f}")
    print(f"Number of trades on test set: {len(test_trades)}")
    print("-" * 30)

    # 6. 可视化测试集上的结果
    plt.figure(figsize=(18, 10))
    
    # 子图1: 价格和交易点
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(df_test['date'], df_test['close'], label="Close Price", color="black", alpha=0.8)
    
    buy_trades = [t for t in test_trades if "buy" in t[1] or "cover" in t[1]]
    sell_trades = [t for t in test_trades if "sell" in t[1] or "short" in t[1]]
    
    if buy_trades:
        buy_dates, _, buy_prices = zip(*buy_trades)
        ax1.scatter(buy_dates, buy_prices, marker="^", color="green", s=100, label="Buy/Cover", zorder=5)
    if sell_trades:
        sell_dates, _, sell_prices = zip(*sell_trades)
        ax1.scatter(sell_dates, sell_prices, marker="v", color="red", s=100, label="Sell/Short", zorder=5)
        
    ax1.set_title(f"Decision Tree Strategy on Test Set (Profit: {test_profit:.2f})")
    ax1.set_ylabel("Price")
    ax1.legend()
    ax1.grid(True)

    # 子图2: 累计收益曲线
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    # 确保pnl_list和df_test的日期对齐
    if len(test_pnl) == len(df_test):
        ax2.plot(df_test['date'], test_pnl, label="Cumulative PnL", color="blue")
    else:
        print("Warning: PnL curve length mismatch, skipping plot.")
    
    ax2.axhline(0, ls="--", c="gray")
    ax2.set_title("Cumulative Profit/Loss (PnL) Curve")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Cumulative PnL")
    ax2.grid(True)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

