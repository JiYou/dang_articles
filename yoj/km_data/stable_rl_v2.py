import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import Env, spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# =================== 策略回测函数 ===================
def moving_average(series, n):
    return series.rolling(window=n, min_periods=1).mean()

def crossover(series1, series2):
    s1_shifted = series1.shift(1)
    s2_shifted = series2.shift(1)
    return (s1_shifted < s2_shifted) & (series1 >= series2)

def crossunder(series1, series2):
    s1_shifted = series1.shift(1)
    s2_shifted = series2.shift(1)
    return (s1_shifted > s2_shifted) & (series1 <= series2)

def strategy_profit(df, length0, p1, p2, p3):
    """计算给定参数组合在数据集上的总利润"""
    if len(df) < 2:
        return 0.0
        
    close = df['close']
    openp = df['open']

    avg1 = moving_average(close, length0)
    avg2 = moving_average(close, length0 + p1)
    avg3 = moving_average(close, length0 + p1 + p2)
    avg4 = moving_average(close, length0 + p1 + p2 + p3)

    buy_signal = (
        crossover(avg1, avg2) | crossover(avg1, avg3) | crossover(avg1, avg4) |
        crossover(avg2, avg3) | crossover(avg2, avg4) | crossover(avg3, avg4)
    )
    sell_signal = (
        crossunder(avg1, avg2) | crossunder(avg1, avg3) | crossunder(avg1, avg4) |
        crossunder(avg2, avg3) | crossunder(avg2, avg4) | crossunder(avg3, avg4)
    )

    position = 0
    entry_price = 0
    profit = 0.0

    for i in range(1, len(df)):
        if buy_signal.iloc[i]:
            if position == -1:
                profit += entry_price - openp.iloc[i]
                position = 0
            if position == 0:
                entry_price = openp.iloc[i]
                position = 1
        elif sell_signal.iloc[i]:
            if position == 1:
                profit += openp.iloc[i] - entry_price
                position = 0
            if position == 0:
                entry_price = openp.iloc[i]
                position = -1

    # 在数据段结束时平仓
    if position == 1:
        profit += close.iloc[-1] - entry_price
    elif position == -1:
        profit += entry_price - close.iloc[-1]

    return profit

# =================== 自定义 Gym 环境 ===================
class DualMAEnv(Env):
    metadata = {'render_modes': []}

    def __init__(self, df):
        super(DualMAEnv, self).__init__()
        self.df = df
        self.length0_space = range(3, 16)
        self.p1_space = range(2, 16)
        self.p2_space = range(2, 16)
        self.p3_space = range(2, 16)
        self.actions = []
        for l0 in self.length0_space:
            for p1 in self.p1_space:
                for p2 in self.p2_space:
                    for p3 in self.p3_space:
                        self.actions.append((l0, p1, p2, p3))
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        observation = np.array([0.0], dtype=np.float32)
        info = {}
        return observation, info

    def step(self, action):
        params = self.actions[action]
        reward = strategy_profit(self.df, *params)
        terminated = True
        truncated = False
        info = {"params": params, "profit": reward}
        observation = np.array([0.0], dtype=np.float32)
        return observation, reward, terminated, truncated, info

# =================== Callback 绘制训练曲线 ===================
class RewardLogger(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLogger, self).__init__(verbose)
        self.rewards = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if len(self.locals["rewards"]) > 0:
            self.rewards.append(np.mean(self.locals["rewards"]))

# =================== 主程序 ===================
def main():
    if len(sys.argv) < 3:
        print("Usage: python your_script_name.py data.csv split_ratio")
        print("Example: python your_script_name.py a9888.DCE1d.txt 0.6")
        sys.exit(1)
        
    filename = sys.argv[1]
    try:
        split_ratio = float(sys.argv[2])
        if not (0 < split_ratio < 1):
            raise ValueError("Split ratio must be between 0 and 1.")
    except ValueError as e:
        print(f"Error: Invalid split_ratio. {e}")
        sys.exit(1)

    # 1. 加载并分割数据
    df_full = pd.read_csv(filename)
    if 'date' in df_full.columns:
        df_full['date'] = pd.to_datetime(df_full['date'])

    split_index = int(len(df_full) * split_ratio)
    df_train = df_full.iloc[:split_index].copy()
    df_test = df_full.iloc[split_index:].copy().reset_index(drop=True)

    print(f"Total data points: {len(df_full)}")
    print(f"Training data points: {len(df_train)} (first {split_ratio*100:.0f}%)")
    print(f"Testing data points: {len(df_test)} (last {(1-split_ratio)*100:.0f}%)")
    print("-" * 30)

    # 2. 在训练集上进行强化学习
    print("Starting RL training on the training set...")
    env_train = DualMAEnv(df_train)
    callback = RewardLogger()
    model = PPO("MlpPolicy", env_train, verbose=0)
    model.learn(total_timesteps=200, callback=callback)
    print("Training finished.")
    print("-" * 30)

    # 3. 获取最优参数并评估在训练集上的表现
    obs, _ = env_train.reset()
    action, _ = model.predict(obs, deterministic=True)
    best_params = env_train.actions[action]
    train_profit = strategy_profit(df_train, *best_params)
    
    print("--- In-Sample Results (Training Set) ---")
    print(f"Optimal parameters found: {best_params}")
    print(f"Profit on training set: {train_profit:.2f}")
    print("-" * 30)

    # 4. 在测试集上验证最优参数的表现
    test_profit = strategy_profit(df_test, *best_params)
    print("--- Out-of-Sample Results (Test Set) ---")
    print(f"Applying parameters {best_params} to the test set.")
    print(f"Profit on test set: {test_profit:.2f}")
    print("-" * 30)

    # 5. 在完整数据集上绘制累计收益曲线
    def calculate_cumulative_return(df, length0, p1, p2, p3):
        """计算并返回每日累计收益序列"""
        if len(df) < 2:
            return [0]
            
        close = df['close']
        openp = df['open']
        avg1 = moving_average(close, length0)
        avg2 = moving_average(close, length0 + p1)
        avg3 = moving_average(close, length0 + p1 + p2)
        avg4 = moving_average(close, length0 + p1 + p2 + p3)

        buy_signal = (crossover(avg1, avg2) | crossover(avg1, avg3) | crossover(avg1, avg4) |
                      crossover(avg2, avg3) | crossover(avg2, avg4) | crossover(avg3, avg4))
        sell_signal = (crossunder(avg1, avg2) | crossunder(avg1, avg3) | crossunder(avg1, avg4) |
                       crossunder(avg2, avg3) | crossunder(avg2, avg4) | crossunder(avg3, avg4))

        position = 0
        entry_price = 0
        pnl_list = [0.0]
        total_pnl = 0.0

        for i in range(1, len(df)):
            daily_pnl = 0.0
            # 首先，根据前一天的持仓计算当日价格变动带来的浮动盈亏
            if position == 1:
                daily_pnl = close.iloc[i] - close.iloc[i-1]
            elif position == -1:
                daily_pnl = close.iloc[i-1] - close.iloc[i]

            # 其次，处理当天的交易信号（开盘价成交）
            if buy_signal.iloc[i]:
                if position == -1: # 平空仓
                    daily_pnl += entry_price - openp.iloc[i]
                # 开多仓
                position = 1
                entry_price = openp.iloc[i]
            elif sell_signal.iloc[i]:
                if position == 1: # 平多仓
                    daily_pnl += openp.iloc[i] - entry_price
                # 开空仓
                position = -1
                entry_price = openp.iloc[i]
            
            total_pnl += daily_pnl
            pnl_list.append(total_pnl)
        return pnl_list

    cum_returns_full = calculate_cumulative_return(df_full, *best_params)

    plt.figure(figsize=(15, 7))
    # 绘制累计收益曲线
    if 'date' in df_full.columns:
        plt.plot(df_full['date'], cum_returns_full, label="Cumulative Return")
        x_axis = df_full['date']
    else:
        plt.plot(cum_returns_full, label="Cumulative Return")
        x_axis = df_full.index

    # 绘制训练集和测试集的分割线
    split_point_date = x_axis.iloc[split_index]
    plt.axvline(x=split_point_date, color='r', linestyle='--', label=f'Train/Test Split ({split_ratio*100:.0f}%)')
    
    # 在图上标注训练集和测试集区域
    plt.text(x_axis.iloc[int(split_index * 0.4)], max(cum_returns_full)*0.9, 'Training Set', fontsize=12, color='blue')
    plt.text(x_axis.iloc[int(split_index + (len(df_full)-split_index)*0.4)], max(cum_returns_full)*0.9, 'Test Set', fontsize=12, color='green')

    plt.title(f"Cumulative Return with RL Optimal Params {best_params}")
    plt.xlabel("Date" if 'date' in df_full.columns else "Time Step")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

