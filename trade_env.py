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
            account_stop_loss_ratio=0.10,        # 账户整体止损比例
            account_take_profit_ratio=0.10,      # 账户整体止盈比例
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
        self.total_step = 0
        self.total_reward = 0

    def reset(self, *, seed=None, options=None, initial_data=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.total_reward = 0
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
            if self.account.long_position <= 0 and self.account.short_position <= 0:
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
            reward -= 0.05
        # elif account_order_res is True:
        #     reward += 0.01

        net_worth, old_net_worth, max_net_worth = self.account.update_net_worth(current_price)

        gain_ratio = self.account.get_gain_ratio()
        if gain_ratio >= self.account_take_profit_ratio:
            terminated = True
        elif gain_ratio <= -self.account_stop_loss_ratio:
            terminated = True

        reward += (net_worth - old_net_worth) / self.account.initial_balance * 100

        # # ✅ 本步收益（只在净值上涨时给予） 净值奖励 避免亏损反弹
        # if net_worth > old_net_worth and net_worth > self.account.initial_balance:
        #     reward += (min((net_worth - old_net_worth), (net_worth - self.account.initial_balance))
        #                / self.account.initial_balance * 100)
        # # ✅ 回撤惩罚（只惩罚新增回撤）
        # drawdown = (max_net_worth - net_worth) / max_net_worth if max_net_worth > 0 else 0
        # prev_drawdown = (max_net_worth - old_net_worth) / max_net_worth if max_net_worth > 0 else 0
        # dd_delta = drawdown - prev_drawdown
        # if dd_delta > 0:
        #     reward -= dd_delta * 100

        if not self.live_mode and self.current_step >= len(self.data_array) - 1:
            terminated = True

        truncated = False
        info = {
            'net_worth': self.account.net_worth,
            'action': action,
            'reward': reward,
            'total_reward': self.total_reward
        }
        if terminated: #or self.total_step-self.last_print_step > 10000
            print(info)
            # self.last_print_step = self.total_step
        self.total_reward += reward
        self.current_step += 1
        self.total_step += 1
        return self._get_observation(), reward, terminated, truncated, info

    def render(self):
        """渲染当前环境状态"""
        print(f"步数: {self.current_step}, 余额: {self.account.balance:.2f}, "
              f"多头: {self.account.long_position:.6f}, 空头: {self.account.short_position:.6f}, "
              f"净值: {self.account.net_worth:.2f}")


