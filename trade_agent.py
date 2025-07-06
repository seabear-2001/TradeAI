import os
import pandas as pd
from sb3_contrib import QRDQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

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
        if policy_kwargs:
            model_kwargs['policy_kwargs'] = policy_kwargs
        model_kwargs['device'] = device
        model_kwargs['verbose'] = 1

        # 提取 gradient_steps 参数，避免冲突
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
        # 直接调用 stable_baselines3 的 save 方法，保存全部状态
        model.save(path)
        print(f"[模型已保存至 {path}]")

    @staticmethod
    def load_model(path, env=None, device="cpu"):
        if not os.path.exists(path):
            print(f"模型文件不存在: {path}")
            return None
        try:
            model = QRDQN.load(path, env=env, device=device)
            print(f"[模型已加载 {path}]")
            return model
        except Exception as e:
            print(f"加载模型失败: {e}")
            return None

    def train_model(
        self,
        model_save_path,
        df,
        model_load_path=None,
        models_backup_path=".",
        tech_indicator_list=None,
        model_kwargs=None,
        policy_kwargs=None,
        eval_freq=1_000_000,
        single_step_num=3,
        num_envs=1,
        device="cpu",
    ):
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        check_timestamp_consistency(df)

        env = make_vec_env(df, tech_indicator_list, num_envs)
        model = None
        if model_load_path:
            model = self.load_model(path=model_load_path, env=env, device=device)

        if model is None:
            model = self.get_model(model_kwargs, policy_kwargs, env, device)
            print(f"[新模型创建完成]")
        else:
            print(f"[模型 {model_save_path}] 继续训练")
            model.set_env(env)

        total_timesteps = len(df) * single_step_num

        try:
            model.learn(
                total_timesteps=int(total_timesteps),
                progress_bar=True,
                reset_num_timesteps=False,
                callback=CheckpointCallback(
                    save_freq=eval_freq,
                    save_path=models_backup_path,
                    name_prefix="qrdqn_model"
                )
            )
        except KeyboardInterrupt:
            print("训练被手动终止，开始保存模型...")

        self.save_model(model_save_path, model)
        print(f"[模型 {model_save_path}] 训练完成")

