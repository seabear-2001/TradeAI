import pandas as pd
import torch
from trade_agent import TradeAgent, check_timestamp_consistency
from trade_env import TradeEnv
from main import all_indicator_names

device = 'cpu'

model_path = './OKX-BTC-USDT-SWAP-1s.zip'
data_path = "./OKX-BTC-USDT-SWAP-1s-features.csv"


# 训练集比例
TRAIN_RATIO = 0.8

def main():
    print(f"设备: {device}")

    # 读取预处理好的特征数据CSV
    base_data = pd.read_csv(data_path)
    print(f"原始数据长度: {len(base_data)}")

    # 分割数据集，取最后20%作为测试集
    split_index = int(len(base_data) * TRAIN_RATIO)
    df = base_data.iloc[split_index:].copy()
    print(f"测试数据长度: {len(df)}")

    # 检查时间戳连贯性
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    check_timestamp_consistency(df)

    # 实例化交易代理
    agent = TradeAgent()

    env = TradeEnv(df = df,tech_indicator_list=all_indicator_names)

    # 加载模型并赋予环境
    model = agent.load_model(
        path=model_path,
        env=env,
        device=device,
    )
    if model is None:
        print("❌ 模型加载失败")
        return

    # 模拟一轮预测（无探索），例如 run 10 episodes
    obs, _ = env.reset()
    episode_rewards = []
    for _ in range(1):
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated , info = env.step(action)
            total_reward += reward
            print(info)

        episode_rewards.append(total_reward)
        obs, _ = env.reset()
        print(total_reward)

    print("🎯 测试集每轮回报:", episode_rewards)
    print("📊 平均回报:", sum(episode_rewards) / len(episode_rewards))

if __name__ == "__main__":
    main()
