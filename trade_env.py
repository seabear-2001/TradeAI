import gymnasium
from gymnasium import spaces
import numpy as np
from trade_account import TradeAccount


class TradeEnv(gymnasium.Env):
    """加密货币交易环境，状态是过去 seq_len 个时间步的行情+账户信息序列"""

    def __init__(
            self,
            df=None,  # 训练数据 DataFrame
            tech_indicator_list=None,  # 技术指标列名列表
            account=None,  # 账户对象实例
            lstm_seq_len=100,  # 序列长度，LSTM 输入需要
    ):
        super().__init__()

        self.lstm_seq_len = lstm_seq_len

        # 初始化账户，如果无则创建默认账户
        self.account = account if account is not None else TradeAccount()

        # 需要用到的行情字段
        self.df_fields = ['open', 'high', 'low', 'close', 'volume'] + (tech_indicator_list or [])

        # 转换成 numpy array，方便索引
        if df is not None:
            self.data_array = df[self.df_fields].values.astype(np.float32)
        else:
            self.data_array = np.empty((0, len(self.df_fields)), dtype=np.float32)

        # 动作空间，5个离散动作
        self.action_space = spaces.Discrete(5)

        # 账户状态的维度，这里假设 get_account_state() 返回长度为5的数组（余额、仓位等）
        account_state_dim = len(self.account.get_account_state())

        # 状态空间：形状是 (seq_len, feature_dim)，每个时间步是行情+账户状态拼接向量
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.lstm_seq_len, len(self.df_fields) + account_state_dim),
            dtype=np.float32
        )

        # 初始化环境状态计数器
        self.current_step = self.lstm_seq_len - 1  # 从 seq_len-1 开始，确保有完整序列
        self.total_reward = 0
        self.total_step = 0

    def reset(self, *, seed=None, options=None, initial_data=None):
        """
        环境重置，清空账户，初始化步数和奖励
        返回初始状态序列和 info
        """
        super().reset(seed=seed)
        self.account.reset()
        self.total_reward = 0
        self.total_step = 0

        # 起始步数重置为 seq_len-1，保证初始状态序列完整
        self.current_step = self.lstm_seq_len - 1

        # 检查数据是否足够
        assert len(self.data_array) >= self.lstm_seq_len, \
            f"数据长度不足 seq_len={self.lstm_seq_len}，当前长度={len(self.data_array)}"

        return self._get_observation(), {}

    def _get_observation(self):
        """
        生成当前时间步的状态序列（seq_len 长度），
        每个时间步拼接行情数据和账户状态
        """
        if self.current_step >= len(self.data_array):
            # 超出数据范围，返回零状态
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        # 计算序列起止索引
        start_idx = self.current_step - self.lstm_seq_len + 1
        end_idx = self.current_step + 1

        obs_seq = []
        for idx in range(start_idx, end_idx):
            # 当前时间步行情数据
            current_data = self.data_array[idx]
            # 当前账户状态
            account_state = self.account.get_account_state()
            # 拼接为一个完整特征向量
            obs = np.append(current_data, account_state)
            obs_seq.append(obs)

        # 转为 np.array，shape = (seq_len, 特征维度)
        return np.array(obs_seq, dtype=np.float32)

    def step(self, action):
        """
        执行动作，计算奖励，更新状态
        返回: next_obs, reward, terminated, truncated, info
        """

        # 当前价格为收盘价
        current_price = self.data_array[self.current_step][3]
        self.account.set_price(current_price)

        reward = 0.0
        terminated = False
        account_order_res = None
        account_reset = False

        # 执行动作逻辑
        if action == 0:
            # 空动作，若无持仓则无效
            if self.account.long_position <= 0 and self.account.short_position <= 0:
                account_order_res = False
        elif action == 1:  # 开多仓
            account_order_res = self.account.open_long()
        elif action == 2:  # 平多仓
            account_order_res = self.account.close_long()
        elif action == 3:  # 开空仓
            account_order_res = self.account.open_short()
        elif action == 4:  # 平空仓
            account_order_res = self.account.close_short()

        # 持仓时空动作小奖励，鼓励持仓操作
        if action == 0 and (self.account.long_position > 0 or self.account.short_position > 0):
            reward += 0.02

        # 无效动作惩罚
        if account_order_res is False:
            reward -= 0.02

        # 平仓成功奖励真实盈利，归一化
        if action in [2, 4] and account_order_res:
            reward += account_order_res / self.account.balance * 100

        # 账户净值变动奖励
        reward += (self.account.net_worth - self.account.old_net_worth) / self.account.balance * 100

        gain_ratio = self.account.get_gain_ratio()

        # 达到盈利或亏损阈值，重置账户并终止回合
        if gain_ratio >= 0.1:
            reward += 2.0
            account_reset = True
        elif gain_ratio <= -0.1:
            reward -= 2.0
            account_reset = True

        # 结束条件：数据到头或账户重置
        if self.current_step >= len(self.data_array) - 1:
            terminated = True

        # 更新环境计数
        self.total_reward += reward
        self.current_step += 1
        self.total_step += 1

        info = {
            "current_step": self.current_step,
            'net_worth': self.account.net_worth,
            'action': action,
            'reward': reward,
            'total_reward': self.total_reward
        }

        if terminated or account_reset:
            print(f"终止信息: {info}")
            self.account.reset()
            self.total_reward = 0

        # 返回 (下一个状态序列, reward, terminated, truncated=False, info)
        return self._get_observation(), reward, terminated, False, info

    def render(self):
        """打印当前环境状态"""
        print(
            f"步数: {self.current_step}, 余额: {self.account.balance:.2f}, "
            f"多仓: {self.account.long_position:.6f}, 空仓: {self.account.short_position:.6f}, "
            f"净值: {self.account.net_worth:.2f}"
        )
