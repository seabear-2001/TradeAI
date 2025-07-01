import numpy as np

class TradeAccount:
    """支持多档拆分持仓的交易账户"""

    def __init__(
            self,
            initial_balance=1000000,
            leverage=100,
            fee_rate=0.0005,
            slots=5  # 拆分仓位档位数量，默认5档
    ):
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.fee_rate = fee_rate
        self.slots = slots

        self.balance = initial_balance
        self.net_worth = initial_balance
        self.max_net_worth = initial_balance

        # 多头和空头仓位数组，存每档仓位数量和均价
        self.long_positions = np.zeros(slots, dtype=np.float64)
        self.long_avg_prices = np.zeros(slots, dtype=np.float64)
        self.short_positions = np.zeros(slots, dtype=np.float64)
        self.short_avg_prices = np.zeros(slots, dtype=np.float64)

        self.trades = []
        self.loss_trade_count = 0
        self.consecutive_loss_count = 0

    def reset(self):
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.long_positions.fill(0)
        self.long_avg_prices.fill(0)
        self.short_positions.fill(0)
        self.short_avg_prices.fill(0)
        self.trades = []
        self.loss_trade_count = 0
        self.consecutive_loss_count = 0

    def open_long(self, slot_idx, current_price, amount):
        """指定档位开多仓"""
        if amount <= 1e-8 or not (0 <= slot_idx < self.slots):
            return False
        fee = amount * current_price * self.fee_rate
        cost = amount * current_price + fee
        if self.balance * self.leverage < cost:
            return False

        old_amount = self.long_positions[slot_idx]
        old_avg = self.long_avg_prices[slot_idx]

        total_cost = old_amount * old_avg + amount * current_price
        new_amount = old_amount + amount
        new_avg = total_cost / new_amount

        self.long_positions[slot_idx] = new_amount
        self.long_avg_prices[slot_idx] = new_avg
        self.balance -= cost / self.leverage

        self.trades.append({
            'type': 'open_long',
            'slot': slot_idx,
            'amount': amount,
            'price': current_price
        })
        return True

    def close_long(self, slot_idx, current_price, amount):
        """指定档位平多仓"""
        if amount <= 1e-8 or not (0 <= slot_idx < self.slots):
            return False
        old_amount = self.long_positions[slot_idx]
        if old_amount < amount:
            return False

        fee = amount * current_price * self.fee_rate
        profit = (current_price - self.long_avg_prices[slot_idx]) * amount
        margin = self.long_avg_prices[slot_idx] * amount / self.leverage

        self.balance += margin + profit - fee
        self.long_positions[slot_idx] -= amount
        if self.long_positions[slot_idx] < 1e-8:
            self.long_positions[slot_idx] = 0.0
            self.long_avg_prices[slot_idx] = 0.0

        self.trades.append({
            'type': 'close_long',
            'slot': slot_idx,
            'amount': amount,
            'price': current_price
        })

        if profit < 0:
            self.loss_trade_count += 1
            self.consecutive_loss_count += 1
        else:
            self.consecutive_loss_count = 0

        return profit

    def open_short(self, slot_idx, current_price, amount):
        """指定档位开空仓"""
        if amount <= 1e-8 or not (0 <= slot_idx < self.slots):
            return False
        fee = amount * current_price * self.fee_rate
        cost = fee
        if self.balance * self.leverage < cost:
            return False

        old_amount = self.short_positions[slot_idx]
        old_avg = self.short_avg_prices[slot_idx]

        total_cost = old_amount * old_avg + amount * current_price
        new_amount = old_amount + amount
        new_avg = total_cost / new_amount

        self.short_positions[slot_idx] = new_amount
        self.short_avg_prices[slot_idx] = new_avg
        self.balance -= cost / self.leverage

        self.trades.append({
            'type': 'open_short',
            'slot': slot_idx,
            'amount': amount,
            'price': current_price
        })
        return True

    def close_short(self, slot_idx, current_price, amount):
        """指定档位平空仓"""
        if amount <= 1e-8 or not (0 <= slot_idx < self.slots):
            return False
        old_amount = self.short_positions[slot_idx]
        if old_amount < amount:
            return False

        fee = amount * current_price * self.fee_rate
        profit = (self.short_avg_prices[slot_idx] - current_price) * amount
        margin = self.short_avg_prices[slot_idx] * amount / self.leverage

        self.balance += margin + profit - fee
        self.short_positions[slot_idx] -= amount
        if self.short_positions[slot_idx] < 1e-8:
            self.short_positions[slot_idx] = 0.0
            self.short_avg_prices[slot_idx] = 0.0

        self.trades.append({
            'type': 'close_short',
            'slot': slot_idx,
            'amount': amount,
            'price': current_price
        })

        if profit < 0:
            self.loss_trade_count += 1
            self.consecutive_loss_count += 1
        else:
            self.consecutive_loss_count = 0

        return profit

    def update_net_worth(self, current_price):
        old_net_worth = self.net_worth
        long_val = np.sum(self.long_positions * (current_price - self.long_avg_prices))
        short_val = np.sum(self.short_positions * (self.short_avg_prices - current_price))
        self.net_worth = self.balance + long_val + short_val
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        return self.net_worth, old_net_worth

    def get_account_state(self):
        # 这里可以改成返回更详细的分仓信息，或者简单返回总仓位和均价
        return np.array([
            self.balance,
            np.sum(self.long_positions),
            np.sum(self.short_positions),
            np.mean(self.long_avg_prices[self.long_positions > 0]) if np.any(self.long_positions > 0) else 0.0,
            np.mean(self.short_avg_prices[self.short_positions > 0]) if np.any(self.short_positions > 0) else 0.0
        ], dtype=np.float32)

    def get_drawdown(self):
        return (self.max_net_worth - self.net_worth) / self.max_net_worth if self.max_net_worth > 0 else 0

    def get_gain_ratio(self):
        return (self.net_worth - self.initial_balance) / self.initial_balance
