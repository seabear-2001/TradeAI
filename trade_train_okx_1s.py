import os
import platform

import pandas as pd
import torch
from trade_agent import TradeAgent

device = 'cuda' if torch.cuda.is_available() else 'cpu'

num_envs = 20

# 模型保存路径
eval_path = '/root/autodl-fs/'
model_path = '/root/autodl-fs/OKX-BTC-USDT-SWAP-1s.pt'
data_path = "/root/autodl-fs/OKX-BTC-USDT-SWAP-1s-features.csv"
system_name = platform.system()
if system_name == "Windows":
    num_envs = 1
    eval_path = './'
    model_path = './OKX-BTC-USDT-SWAP-1s.pt'
    data_path = "./OKX-BTC-USDT-SWAP-1s-features.csv"


# 训练数据集划分比例
TRAIN_RATIO = 0.8

single_step_num = 32 # 每步训练的重复次数
eval_freq = 1_000_000

# QRDQN算法相关超参数配置，参考SB3文档和经验调整
model_kwargs = {
    "learning_rate": 1e-4,            # 学习率，越小越稳定
    "buffer_size": 50_000_000,           # 经验回放池大小，越大越稳定但占内存
    "learning_starts": 20_000,        # 收集多少步后开始训练
    "batch_size": 4096,                # 每次训练采样大小
    "train_freq": 2,                  # 每执行多少步训练一次模型 和 每次训练的更新步数
    "gradient_steps": 1,              # 每次训练的更新步数
    "target_update_interval": 1000,   # 目标网络更新频率
    "exploration_fraction": 0.5,      # epsilon衰减比例，前40%训练是探索
    "exploration_final_eps": 0.05,    # epsilon最终最小值
    "gamma": 0.99,                   # 折扣因子，考虑未来奖励的权重
    "max_grad_norm": 10  # 梯度裁剪
}

# 策略网络结构及激活函数
policy_kwargs = dict(
    net_arch=[512, 512, 256, 128],              # 两层全连接网络，256和128神经元
    activation_fn=torch.nn.ReLU       # 激活函数ReLU
)

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
def main():
    # 读取预处理好的特征数据CSV
    base_data = pd.read_csv(data_path)
    print(f"原始数据长度: {len(base_data)}")

    # 按比例划分训练集
    train_df = base_data.iloc[:int(len(base_data) * TRAIN_RATIO)]

    # 实例化交易代理
    agent = TradeAgent()
    # model , _ = agent.load_model("./OKX-BTC-USDT-SWAP-1s.pt", device=device)
    # 训练模型
    agent.train_model(
        # model=model,
        path=model_path,                    # 模型保存路径
        df=train_df,                       # 训练数据DataFrame
        eval_path=eval_path,              # 模型评估数据保存路径
        eval_freq=eval_freq,              # 模型评估频率
        tech_indicator_list=all_indicator_names,  # 技术指标列表
        single_step_num=single_step_num,  # 每条数据训练多少步
        model_kwargs=model_kwargs,         # QRDQN超参数
        policy_kwargs=policy_kwargs,       # 策略网络参数
        device=device,                    # 设备
        num_envs=num_envs,
    )


if __name__ == "__main__":
    main()
