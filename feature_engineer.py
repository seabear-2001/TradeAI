import pandas as pd
import ta
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

all_indicator_names = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
    "ema_12",
    "ema_26",
    "momentum_10",
    "roc_10",
    "willr_14",
    "stoch_k_14",
    "stoch_d_14",
    "atr_14",
    "trange",
    "obv",
    "vwap"
]

# 指标所需最小窗口长度（不在此字典中的视为0）
indicator_windows = {
    "dx_30": 30,
    "atr_14": 14,
    "rsi_30": 30,
    "cci_30": 30,
    "boll_ub": 20,  # BollingerBands 默认20窗口
    "boll_lb": 20,
    "close_30_sma": 30,
    "close_60_sma": 60,
    "ema_12": 12,
    "ema_26": 26,
    "momentum_10": 10,
    "roc_10": 10,
    "willr_14": 14,
    "stoch_k_14": 14,
    "stoch_d_14": 14,
    "trange": 1,
    "obv": 1,
    "vwap": 1,
    "macd": 26,  # macd默认26窗口
}


def get_max_window(tech_indicator_list):
    max_window = 0
    for ind in tech_indicator_list:
        w = indicator_windows.get(ind, 0)
        if w > max_window:
            max_window = w
    return max_window


def compute_indicator(name, df_dict):
    try:
        close = df_dict['close']
        high = df_dict['high']
        low = df_dict['low']
        volume = df_dict.get('volume', None)
        length = len(close)

        # 数据长度不足，返回全空，避免报错
        required_window = indicator_windows.get(name, 0)
        if length < required_window:
            print(f"[进程] 数据长度{length}不足以计算指标{name}（需要至少{required_window}条）")
            return name, pd.Series([None] * length)

        if name == "macd":
            return name, ta.trend.MACD(close).macd()
        elif name == "boll_ub":
            return name, ta.volatility.BollingerBands(close).bollinger_hband()
        elif name == "boll_lb":
            return name, ta.volatility.BollingerBands(close).bollinger_lband()
        elif name == "rsi_30":
            return name, ta.momentum.RSIIndicator(close, window=30).rsi()
        elif name == "cci_30":
            return name, ta.trend.CCIIndicator(high, low, close, window=30).cci()
        elif name == "dx_30":
            return name, ta.trend.ADXIndicator(high, low, close, window=30).adx()
        elif name == "close_30_sma":
            return name, close.rolling(window=30).mean()
        elif name == "close_60_sma":
            return name, close.rolling(window=60).mean()
        elif name == "ema_12":
            return name, ta.trend.EMAIndicator(close, window=12).ema_indicator()
        elif name == "ema_26":
            return name, ta.trend.EMAIndicator(close, window=26).ema_indicator()
        elif name == "momentum_10":
            return name, close - close.shift(10)
        elif name == "roc_10":
            return name, ta.momentum.ROCIndicator(close, window=10).roc()
        elif name == "willr_14":
            return name, ta.momentum.WilliamsRIndicator(high, low, close, lbp=14).williams_r()
        elif name == "stoch_k_14":
            return name, ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3).stoch()
        elif name == "stoch_d_14":
            return name, ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3).stoch_signal()
        elif name == "atr_14":
            return name, ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
        elif name == "trange":
            return name, ta.volatility.AverageTrueRange(high, low, close, window=1).average_true_range()
        elif name == "obv":
            if volume is not None:
                return name, ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
            else:
                print("[进程] 缺少 volume，无法计算 obv")
                return name, pd.Series([None] * length)
        elif name == "vwap":
            if volume is not None:
                cum_vol = volume.cumsum()
                cum_vol_price = (close * volume).cumsum()
                vwap = cum_vol_price / cum_vol
                return name, vwap
            else:
                print("[进程] 缺少 volume，无法计算 vwap")
                return name, pd.Series([None] * length)
        else:
            print(f"[进程] 未知指标名: {name}")
            return name, pd.Series([None] * length)
    except Exception as e:
        print(f"[进程] 计算指标 {name} 出错: {e}")
        return name, pd.Series([None] * len(close))


def unpack_and_compute(args):
    return compute_indicator(*args)


class FeatureEngineer:
    def __init__(self, tech_indicator_list):
        self.tech_indicator_list = tech_indicator_list

    def preprocess_data(self, df: pd.DataFrame, use_multiprocessing: bool = True) -> pd.DataFrame:
        df = df.copy()

        df_dict = {
            'close': df['close'],
            'high': df['high'],
            'low': df['low'],
        }
        if 'volume' in df.columns:
            df_dict['volume'] = df['volume']
        else:
            print("[FeatureEngineer] 警告：输入数据缺少 volume 列，部分指标无法计算")

        task_args = [(name, df_dict) for name in self.tech_indicator_list]

        if use_multiprocessing:
            with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                results = list(executor.map(unpack_and_compute, task_args))
        else:
            results = [compute_indicator(name, df_dict) for name in self.tech_indicator_list]

        for name, series in results:
            df[name] = series.values

        return df.ffill().bfill()

    def preprocess_latest_row(self, df: pd.DataFrame) -> pd.DataFrame:
        window_size = get_max_window(self.tech_indicator_list)
        window_size = min(window_size, len(df))

        df_tail = df.tail(window_size).reset_index(drop=True)

        df_dict = {
            'close': df_tail['close'],
            'high': df_tail['high'],
            'low': df_tail['low'],
        }
        if 'volume' in df_tail.columns:
            df_dict['volume'] = df_tail['volume']
        else:
            print("[FeatureEngineer] 警告：缺少 volume 列，部分指标无法计算")

        results = [compute_indicator(name, df_dict) for name in self.tech_indicator_list]

        latest_features = {}
        for name, series in results:
            latest_features[name] = series.iloc[-1]

        df_new = df.copy()

        # 新行先全部赋 None
        new_row = {col: None for col in df_new.columns}
        # 赋指标值
        new_row.update(latest_features)

        # 把新行转为DataFrame
        new_row_df = pd.DataFrame([new_row])

        # 过滤掉全是NA的列，避免 FutureWarning
        new_row_df = new_row_df.dropna(axis=1, how='all')

        # 拼接
        df_new = pd.concat([df_new, new_row_df], ignore_index=True)

        # 赋基本行情数据（close, high, low, volume）到最新行，防止它们是 None
        for col in ['close', 'high', 'low', 'volume']:
            if col in df_tail.columns:
                df_new.at[df_new.index[-1], col] = df_tail.at[df_tail.index[-1], col]

        # 前后填充缺失值
        return df_new.ffill().bfill()

