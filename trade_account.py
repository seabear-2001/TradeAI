import numpy as np


class TradeAccount:
    """管理交易账户的状态和操作"""

    def __init__(
            self,
            initial_balance=1000000,         # 初始账户资金
            leverage=100,                    # 杠杆倍数，用于计算保证金
            fee_rate=0.0000,                  # 交易手续费率
            max_position_ratio= 0.2
    ):
        # 账户参数
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.fee_rate = fee_rate

        # 账户状态
        self.balance = initial_balance  # 当前余额
        self.net_worth = initial_balance  # 当前净值
        self.max_net_worth = initial_balance  # 历史最大净值
        self.max_position_ratio = max_position_ratio
        self.long_position = 0.0  # 多头持仓量
        self.long_ave_price = 0.0  # 多头平均持仓价
        self.short_position = 0.0  # 空头持仓量
        self.short_ave_price = 0.0  # 空头平均持仓价

    def reset(self):
        """重置账户状态到初始值"""
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.long_position = 0.0
        self.long_ave_price = 0.0
        self.short_position = 0.0
        self.short_ave_price = 0.0

    def open_long(self, current_price):
        """按指定数量开多头仓位"""
        max_position_amount = self.initial_balance / current_price * self.max_position_ratio
        amount = max_position_amount
        fee = amount * current_price * self.fee_rate * 0 # 取消开仓手续费
        cost = amount * current_price / self.leverage + fee
        if self.long_position > 0:
            return False
        if self.balance >= cost and amount > 0:
            total_cost = self.long_position * self.long_ave_price + amount * current_price
            self.long_position += amount
            self.long_ave_price = total_cost / self.long_position
            self.balance -= cost
            return True
        return False

    def close_long(self, current_price):
        """按指定数量平多头仓位"""
        if self.long_position <= 0:
            return False

        actual_amount = self.long_position
        fee = actual_amount * current_price * self.fee_rate * 2 # 加上开仓手续费
        profit = (current_price - self.long_ave_price) * actual_amount
        margin = self.long_ave_price * actual_amount / self.leverage

        self.balance += margin + profit - fee
        self.long_position -= actual_amount
        if self.long_position < 0:
            self.long_ave_price = 0.0
        return True

    def open_short(self, current_price):
        """按指定数量开空头仓位"""
        max_position_amount = self.initial_balance / current_price * self.max_position_ratio
        amount = max_position_amount
        fee = amount * current_price * self.fee_rate * 0 # 取消开仓手续费
        cost = amount * current_price / self.leverage + fee
        if self.balance >= cost and amount > 0:
            total_cost = self.short_position * self.short_ave_price + amount * current_price
            self.short_position += amount
            self.short_ave_price = total_cost / self.short_position
            self.balance -= cost
            return True
        return False

    def close_short(self, current_price):
        """按指定数量平空头仓位"""
        if self.short_position <= 0:
            return False

        actual_amount = self.short_position
        fee = actual_amount * current_price * self.fee_rate * 2 # 添加开仓手续费
        profit = (self.short_ave_price - current_price) * actual_amount
        margin = self.short_ave_price * actual_amount / self.leverage

        self.balance += margin + profit - fee
        self.short_position -= actual_amount
        if self.short_position < 0:
            self.short_ave_price = 0.0
        return True


    def update_net_worth(self, current_price):
        """更新账户净值"""
        old_net_worth = self.net_worth
        self.net_worth = self.balance + \
                         self.long_position * (current_price - self.long_ave_price) + \
                         self.short_position * (self.short_ave_price - current_price)
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        return self.net_worth, old_net_worth, self.max_net_worth

    def get_account_state(self):
        """获取账户状态用于观察空间"""
        return np.array([self.balance, self.long_position, self.short_position, self.long_ave_price, self.short_ave_price], dtype=np.float32)

    def get_drawdown(self):
        """计算当前回撤"""
        return (self.max_net_worth - self.net_worth) / self.max_net_worth

    def get_gain_ratio(self):
        """计算账户盈亏比例"""
        return (self.net_worth - self.initial_balance) / self.initial_balance
