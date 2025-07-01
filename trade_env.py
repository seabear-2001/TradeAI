import gymnasium
from gymnasium import spaces

from trade_account import TradeAccount

import numpy as np


class TradeEnv(gymnasium.Env):
    """加密货币交易环境，使用独立的TradingAccount管理账户"""

    def __init__(
            self,
            live_mode=False,                    # 是否实盘模式
            df=None,                            # 训练数据，DataFrame 格式
            tech_indicator_list=None,           # 技术指标列名列表
            account=None,                       # 交易账户实例
            max_position_ratio=0.2,             # 单向最大持仓量占初始本金比例
            account_stop_loss_ratio=0.1,        # 账户整体止损比例
            account_take_profit_ratio=0.2,      # 账户整体止盈比例
    ):
        super().__init__()

        # 参数记录
        self.live_mode = live_mode
        self.max_position_ratio = max_position_ratio
        self.account_stop_loss_ratio = account_stop_loss_ratio
        self.account_take_profit_ratio = account_take_profit_ratio

        # 账户实例
        self.account = account if account is not None else TradeAccount()

        # 数据字段
        self.df_fields = ['open', 'high', 'low', 'close', 'volume'] + (tech_indicator_list or [])
        if df is not None:
            self.data_array = df[self.df_fields].values.astype(np.float32)
        else:
            self.data_array = np.empty((0, len(self.df_fields)), dtype=np.float32)

        self.action_space = spaces.Discrete(21)  # 0~20


        # 状态空间：行情数据 + 账户状态（余额、多空仓位）
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(self.df_fields) + 5,), dtype=np.float32
        )

        # 初始化状态
        self.current_step = 0

    def reset(self, *, seed=None, options=None, initial_data=None):
        super().reset(seed=seed)
        self.current_step = 0

        # 实盘模式数据传入
        if self.live_mode:
            if initial_data is not None:
                # 只保留数值列，避免包含时间戳字符串导致转换失败
                numeric_df = initial_data[self.df_fields].copy()
                self.data_array = numeric_df.values.astype(np.float32)
            else:
                self.data_array = np.empty((0, len(self.df_fields)), dtype=np.float32)
        else:
            self.account.reset()
            assert self.data_array is not None and len(self.data_array) > 0, "训练模式必须提供数据"

        return self._get_observation(), {}

    def _get_observation(self):
        """获取当前观察值（行情 + 账户状态）"""

        if self.current_step >= len(self.data_array):
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        current_data = self.data_array[self.current_step]
        return np.append(current_data, self.account.get_account_state()).astype(np.float32)

    def step(self, action):
        current_price = self.data_array[self.current_step][3]
        max_position_amount = self.account.initial_balance / current_price * self.max_position_ratio
        base_amount = max_position_amount * self.account.leverage / self.account.slots  # 每档大小

        reward = 0.0
        efficient = True
        terminated = False

        if action == 0:
            if np.sum(self.account.long_positions) <= 0 and np.sum(self.account.short_positions) <= 0:
                reward -= 0.01
        elif 1 <= action <= 5:  # 开多档位
            idx = action - 1
            if not self.account.open_long(idx, current_price, base_amount):
                efficient = False
        elif 6 <= action <= 10:  # 平多档位
            idx = action - 6
            amount = self.account.long_positions[idx]
            if amount > 0:
                res = self.account.close_long(idx, current_price, amount)
                if res is False:
                    efficient = False
                else:
                    reward += res / self.account.initial_balance / 100
            else:
                efficient = False
        elif 11 <= action <= 15:  # 开空档位
            idx = action - 11
            if not self.account.open_short(idx, current_price, base_amount):
                efficient = False
        elif 16 <= action <= 20:  # 平空档位
            idx = action - 16
            amount = self.account.short_positions[idx]
            if amount > 0:
                res = self.account.close_short(idx, current_price, amount)
                if res is False:
                    efficient = False
                else:
                    reward += res / self.account.initial_balance / 100
            else:
                efficient = False
        else:
            efficient = False

        # 后续净值更新、回撤、盈亏、止盈止损等逻辑保持不变
        net_worth, old_net_worth = self.account.update_net_worth(current_price)
        net_worth_change = (net_worth - old_net_worth) / self.account.initial_balance
        reward += net_worth_change / self.account.initial_balance
        reward += self.account.get_gain_ratio()
        reward -= self.account.get_drawdown()

        if not efficient:
            reward -= 0.01

        gain_ratio = self.account.get_gain_ratio()
        if gain_ratio >= self.account_take_profit_ratio:
            reward += 0.01
            terminated = True
        elif gain_ratio <= -self.account_stop_loss_ratio:
            reward -= 0.01
            terminated = True

        if not self.live_mode and self.current_step >= len(self.data_array) - 1:
            terminated = True

        truncated = False
        info = {
            'net_worth': self.account.net_worth,
            'action': action,
            'ratio': self.account.get_gain_ratio(),
            'reward': reward
        }

        self.current_step += 1
        return self._get_observation(), reward, terminated, truncated, info

    def render(self):
        """渲染当前环境状态"""
        print(f"步数: {self.current_step}, 余额: {self.account.balance:.2f}, "
              f"多头: {self.account.long_position:.6f}, 空头: {self.account.short_position:.6f}, "
              f"净值: {self.account.net_worth:.2f}")


def check_future_outcome(data_array, current_step, entry_price, direction):
    """
    判断在未来行情中，是否先止盈或止损，或最终是盈利还是亏损
    direction: "long" 或 "short"
    """
    tp_rate = 1.005  # 止盈0.5%
    sl_rate = 0.997  # 止损0.3%
    future_data = data_array[current_step + 1:]
    outcome = "no_target"

    for step_data in future_data:
        high = step_data[1]
        low = step_data[2]
        if direction == "long":
            if high >= entry_price * tp_rate:
                outcome = "盈利"
                break
            if low <= entry_price * sl_rate:
                outcome = "亏损"
                break
        elif direction == "short":
            if low <= entry_price * (2 - tp_rate):  # 等价于 *0.95
                outcome = "盈利"
                break
            if high >= entry_price * (2 - sl_rate):  # 等价于 *1.03
                outcome = "亏损"
                break
    return outcome
