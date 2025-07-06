import os
import pandas as pd
from sb3_contrib import QRDQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

import LSTMFeatureExtractor
from trade_env import TradeEnv


def check_timestamp_consistency(df: pd.DataFrame, time_col: str = 'timestamp') -> None:
    if time_col not in df.columns:
        raise ValueError(f"列 '{time_col}' 不存在")
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        raise TypeError(f"列 '{time_col}' 不是 datetime 类型")

    df_sorted = df.sort_values(by=time_col).reset_index(drop=True)
    time_diffs = df_sorted[time_col].diff().dropna()
    expected_diff = time_diffs.iloc[0]
    inconsistent = time_diffs != expected_diff

    if inconsistent.any():
        raise ValueError(f"❌ 时间间隔不一致，共 {inconsistent.sum()} 条异常记录")
    print("✅ 时间戳连贯，间隔一致")


def make_single_env(df, tech_indicator_list):
    def _init():
        return TradeEnv(df=df.copy(), tech_indicator_list=tech_indicator_list)
    return _init


def make_vec_env(df, tech_indicator_list, num_envs):
    env_fns = [make_single_env(df, tech_indicator_list) for _ in range(num_envs)]
    return SubprocVecEnv(env_fns)


class TradeAgent:
    def __init__(self):
        pass

    @staticmethod
    def get_model(model_kwargs=None, policy_kwargs=None, env=None, device="cpu"):
        model_kwargs = model_kwargs or {}

        # ✅ 设置 LSTM 为特征提取器
        if policy_kwargs is None:
            policy_kwargs = {}
        policy_kwargs['features_extractor_class'] = LSTMFeatureExtractor
        policy_kwargs['features_extractor_kwargs'] = dict(
            lstm_hidden_size=64
        )

        model_kwargs['policy_kwargs'] = policy_kwargs
        model_kwargs['device'] = device
        model_kwargs['verbose'] = 1

        gradient_steps = model_kwargs.pop("gradient_steps", 1)

        model = QRDQN(
            "MlpPolicy",
            env,
            gradient_steps=gradient_steps,
            **model_kwargs
        )
        return model

    @staticmethod
    def save_model(path, model):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model.save(path)
        print(f"[模型已保存至 {path}]")

    @staticmethod
    def load_model(path, env=None, device="cpu", custom_objects=None):
        if not os.path.exists(path):
            print(f"模型文件不存在: {path}")
            return None

        model = QRDQN.load(path, env=env, device=device, custom_objects=custom_objects)
        print(f"[模型已加载 {path}]")
        return model
    def train_model(
        self,
        model_save_path,
        df,
        num_chunks=1,
        model_load_path=None,
        models_backup_path=".",
        tech_indicator_list=None,
        model_kwargs=None,
        custom_objects=None,
        policy_kwargs=None,
        eval_freq=1_000_000,
        single_step_num=3,
        num_envs=1,
        device="cpu",
    ):
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        check_timestamp_consistency(df)

        # 计算每段数据大小
        total_len = len(df)
        chunk_size = total_len // num_chunks

        current_model_path = model_load_path

        for i in range(num_chunks):
            start_idx = i * chunk_size
            # 最后一段包含剩余所有数据
            end_idx = (i + 1) * chunk_size if i < num_chunks - 1 else total_len
            chunk_df = df.iloc[start_idx:end_idx]

            print(f"\n=== 开始第 {i+1} / {num_chunks} 段训练，数据范围: {start_idx} ~ {end_idx} 共 {len(chunk_df)} 条 ===")

            env = make_vec_env(chunk_df, tech_indicator_list, num_envs)

            if current_model_path is not None:
                print(f"加载已有模型 {current_model_path} 进行增量训练")
                model = self.load_model(path=current_model_path, env=env, device=device, custom_objects=custom_objects)
            else:
                model = self.get_model(model_kwargs, policy_kwargs, env, device)
                print("[新模型创建完成]")

            if model is None:
                print(f"模型加载失败，重新创建新模型")
                model = self.get_model(model_kwargs, policy_kwargs, env, device)

            model.set_env(env)

            total_timesteps = len(chunk_df) * single_step_num

            model.learn(
                total_timesteps=int(total_timesteps),
                progress_bar=True,
                reset_num_timesteps=False,
                callback=CheckpointCallback(
                    save_freq=eval_freq,
                    save_path=models_backup_path,
                    name_prefix=f"qrdqn_model_chunk{i+1}"
                )
            )

            # 保存当前段训练后的模型
            save_path = f"{os.path.splitext(model_save_path)[0]}_chunk{i+1}{os.path.splitext(model_save_path)[1]}"
            self.save_model(save_path, model)
            print(f"第 {i+1} 段训练完成，模型保存至 {save_path}")

            # 下一段训练从这个模型开始
            current_model_path = save_path

        print("所有分段训练完成。")