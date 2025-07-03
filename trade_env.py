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
            account_stop_loss_ratio=0.1,        # 账户整体止损比例
            account_take_profit_ratio=0.2,      # 账户整体止盈比例
    ):
        super().__init__()

        # 参数记录
        self.live_mode = live_mode
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

        self.action_space = spaces.Discrete(5)  # 0~4


        # 状态空间：行情数据 + 账户状态（余额、多空仓位）
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(self.df_fields) + 5,), dtype=np.float32
        )

        # 初始化状态
        self.current_step = 0
        self.terminated_count = 0
        self.last_print_step = 0

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

        reward = 0.0
        terminated = False
        account_order_res = None

        if action == 0:
            if np.sum(self.account.long_position) <= 0 and np.sum(self.account.short_position) <= 0:
                account_order_res = False
        elif action == 1:  # 开多档位
            account_order_res = self.account.open_long(current_price)
        elif action == 2:  # 平多档位
            account_order_res = self.account.close_long(current_price)
        elif action == 3:  # 开空档位
            account_order_res = self.account.open_short(current_price)
        elif action == 4:  # 平空档位
            account_order_res = self.account.close_short(current_price)

        if account_order_res is False:
            reward -= 0.01

        gain_ratio = self.account.get_gain_ratio()
        if gain_ratio >= self.account_take_profit_ratio:
            terminated = True
        elif gain_ratio <= -self.account_stop_loss_ratio:
            terminated = True

        # 后续净值更新、回撤、盈亏、止盈止损等逻辑保持不变
        net_worth, old_net_worth, max_net_worth = self.account.update_net_worth(current_price)
        reward += (net_worth - old_net_worth) / self.account.initial_balance * 100  # 本步收益
        reward -= ((max_net_worth - net_worth) / max_net_worth if max_net_worth > 0 else 0) * 20 # 本步回撤

        if not self.live_mode and self.current_step >= len(self.data_array) - 1:
            terminated = True

        truncated = False
        info = {
            'net_worth': self.account.net_worth,
            'action': action,
            'ratio': self.account.get_gain_ratio(),
            'reward': reward
        }
        self.last_print_step = 0
        if self.current_step - self.last_print_step >= 1000 or terminated:
            print(info)
            self.last_print_step = self.current_step

        self.current_step += 1
        return self._get_observation(), reward, terminated, truncated, info

    def render(self):
        """渲染当前环境状态"""
        print(f"步数: {self.current_step}, 余额: {self.account.balance:.2f}, "
              f"多头: {self.account.long_position:.6f}, 空头: {self.account.short_position:.6f}, "
              f"净值: {self.account.net_worth:.2f}")


