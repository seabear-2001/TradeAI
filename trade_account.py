import numpy as np


class TradeAccount:
    def __init__(
            self,
            initial_balance=1000000,
            leverage=100,
            fee_rate=0.0005,
            max_position_ratio=0.2
    ):
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.fee_rate = fee_rate
        self.max_position_ratio = max_position_ratio
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.long_position = 0.0
        self.long_ave_price = 0.0
        self.short_position = 0.0
        self.short_ave_price = 0.0

    def open_long(self, current_price):
        """开多头仓位（动态仓位控制）"""
        # 动态计算最大仓位
        max_amount = (self.net_worth * self.max_position_ratio) / current_price
        amount = max_amount

        # 计算实际需要资金
        trade_value = amount * current_price
        fee = trade_value * self.fee_rate
        margin_required = trade_value / self.leverage
        total_cost = margin_required + fee

        # 检查开仓条件
        if (
                self.balance >= total_cost
                and amount > 0
                and self.short_position == 0  # 无空头持仓
        ):
            # 更新持仓
            if self.long_position == 0:
                self.long_ave_price = current_price
            else:
                total_cost = self.long_position * self.long_ave_price + amount * current_price
                total_amount = self.long_position + amount
                self.long_ave_price = total_cost / total_amount

            self.long_position += amount
            self.balance -= total_cost
            return fee
        return False

    def close_long(self, current_price):
        if self.long_position == 0:
            return False

        # 计算盈亏
        trade_value = self.long_position * current_price
        fee = trade_value * self.fee_rate
        profit = (current_price - self.long_ave_price) * self.long_position
        released_margin = (self.long_position * self.long_ave_price) / self.leverage

        # 更新账户
        self.balance += released_margin + profit - fee
        self.long_position = 0.0
        self.long_ave_price = 0.0  # 重置均价
        return profit

    def open_short(self, current_price):
        """开空头仓位（逻辑同开多）"""
        max_amount = (self.net_worth * self.max_position_ratio) / current_price
        amount = max_amount

        trade_value = amount * current_price
        fee = trade_value * self.fee_rate
        margin_required = trade_value / self.leverage
        total_cost = margin_required + fee

        if (
                self.balance >= total_cost
                and amount > 0
                and self.long_position == 0  # 无多头持仓
        ):
            if self.short_position == 0:
                self.short_ave_price = current_price
            else:
                total_cost = self.short_position * self.short_ave_price + amount * current_price
                total_amount = self.short_position + amount
                self.short_ave_price = total_cost / total_amount

            self.short_position += amount
            self.balance -= total_cost
            return fee
        return False

    def close_short(self, current_price):
        if self.short_position == 0:
            return False

        trade_value = self.short_position * current_price
        fee = trade_value * self.fee_rate
        profit = (self.short_ave_price - current_price) * self.short_position
        released_margin = (self.short_position * self.short_ave_price) / self.leverage

        self.balance += released_margin + profit - fee
        self.short_position = 0.0
        self.short_ave_price = 0.0  # 重置均价
        return profit

    def update_net_worth(self, current_price):
        old_net_worth = self.net_worth

        # 计算浮动盈亏
        long_profit = self.long_position * (current_price - self.long_ave_price) if self.long_position else 0
        short_profit = self.short_position * (self.short_ave_price - current_price) if self.short_position else 0

        self.net_worth = self.balance + long_profit + short_profit
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        return self.net_worth, old_net_worth, self.max_net_worth

    # ... (其他方法保持原样)