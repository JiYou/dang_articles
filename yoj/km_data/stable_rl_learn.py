import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from gym import Env, spaces # 旧版
from gymnasium import Env, spaces # 新版
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env

# =================== 策略回测函数 ===================
def moving_average(series, n):
    return series.rolling(window=n, min_periods=1).mean()

def crossover(series1, series2):
    # 确保索引对齐，避免因NaN产生的警告
    s1_shifted = series1.shift(1)
    s2_shifted = series2.shift(1)
    return (s1_shifted < s2_shifted) & (series1 >= series2)

def crossunder(series1, series2):
    # 确保索引对齐，避免因NaN产生的警告
    s1_shifted = series1.shift(1)
    s2_shifted = series2.shift(1)
    return (s1_shifted > s2_shifted) & (series1 <= series2)

def strategy_profit(df, length0, p1, p2, p3):
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

    # 收盘平仓
    if position == 1:
        profit += close.iloc[-1] - entry_price
    elif position == -1:
        profit += entry_price - close.iloc[-1]

    return profit

# =================== 自定义 Gym 环境 ===================
class DualMAEnv(Env):
    # gymnasium 需要这个元数据
    metadata = {'render_modes': []}

    def __init__(self, df):
        super(DualMAEnv, self).__init__()
        self.df = df
        # 将参数组合离散化，作为动作空间
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
        # 状态空间可以简单设置为单个值占位
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.current_step = 0

    # FIX 1: 更新 reset 方法的签名以匹配 gymnasium API
    def reset(self, seed=None, options=None):
        # 遵循 gymnasium 规范，调用父类的 reset
        super().reset(seed=seed)
        self.current_step = 0
        # reset 方法现在需要返回 (observation, info)
        observation = np.array([0.0], dtype=np.float32)
        info = {}
        return observation, info

    # FIX 2: 更新 step 方法的返回值以匹配 gymnasium API
    def step(self, action):
        params = self.actions[action]
        reward = strategy_profit(self.df, *params)
        
        # 在这个环境中，每一步都是一个完整的 episode，所以它总是 "terminated"
        terminated = True
        truncated = False # 没有被截断
        
        info = {"params": params, "profit": reward}
        
        # step 方法现在需要返回 (observation, reward, terminated, truncated, info)
        observation = np.array([0.0], dtype=np.float32)
        return observation, reward, terminated, truncated, info

# =================== Callback 绘制训练曲线 ===================
class RewardLogger(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLogger, self).__init__(verbose)
        self.rewards = []

    def _on_step(self) -> bool:
        # 在 on_policy_algorithm 中，rollout buffer 在 _on_step 之后收集数据
        # 所以在 _on_rollout_end 中记录奖励更准确
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        # self.locals['rewards'] 包含了最近一次 rollout 中每一步的奖励
        # 在这个特殊环境中，每次 rollout 只有一步，所以直接取均值即可
        if len(self.locals["rewards"]) > 0:
            self.rewards.append(np.mean(self.locals["rewards"]))

# =================== 主程序 ===================
def main():
    if len(sys.argv) < 2:
        print("Usage: python your_script_name.py data.csv")
        sys.exit(1)
    filename = sys.argv[1]
    df = pd.read_csv(filename)
    # 确保日期列是 datetime 类型，方便绘图
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])


    env = DualMAEnv(df)
    # (可选但推荐) 检查环境是否符合 gymnasium 规范
    # check_env(env)

    callback = RewardLogger()
    # 使用 MlpPolicy 因为我们的观察空间是简单的向量
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=200, callback=callback)

    # 绘制训练过程
    plt.figure(figsize=(10,5))
    plt.plot(callback.rewards, label="Average Reward per Rollout")
    plt.xlabel("Rollout")
    plt.ylabel("Profit")
    plt.title("RL Training Process")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 测试最优动作
    # FIX 3: reset() 现在返回两个值
    obs, info = env.reset()
    action, _states = model.predict(obs, deterministic=True)
    best_params = env.actions[action]
    best_profit = strategy_profit(df, *best_params)
    print("最优参数:", best_params)
    print("最优收益:", best_profit)

    # 画最优参数下收益曲线
    length0, p1, p2, p3 = best_params
    close = df['close']
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
    # 修正累计收益计算逻辑
    daily_profit = [0.0] * len(df)
    current_profit = 0.0

    for i in range(1, len(df)):
        # 先计算当日持仓收益
        if position == 1:
            current_profit = df['close'].iloc[i] - df['close'].iloc[i-1]
        elif position == -1:
            current_profit = df['close'].iloc[i-1] - df['close'].iloc[i]
        else:
            current_profit = 0.0
        
        # 再处理交易信号
        if buy_signal.iloc[i]:
            if position == -1: # 平空
                current_profit += entry_price - df['open'].iloc[i]
                position = 0
            if position == 0: # 开多
                entry_price = df['open'].iloc[i]
                position = 1
        elif sell_signal.iloc[i]:
            if position == 1: # 平多
                current_profit += df['open'].iloc[i] - entry_price
                position = 0
            if position == 0: # 开空
                entry_price = df['open'].iloc[i]
                position = -1
        
        daily_profit[i] = current_profit

    # 计算累计收益
    cum_returns = np.cumsum(daily_profit)

    plt.figure(figsize=(12,6))
    # 确保 x 轴是日期
    if 'date' in df.columns:
        plt.plot(df['date'], cum_returns, label="Cumulative Return")
        plt.xlabel("Date")
    else:
        plt.plot(cum_returns, label="Cumulative Return")
        plt.xlabel("Time Step")

    plt.title(f"Cumulative Return with RL Optimal Params {best_params}")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

