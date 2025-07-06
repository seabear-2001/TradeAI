import os
import pandas as pd
from sb3_contrib import QRDQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from gymnasium.wrappers import FlattenObservation

from TransformerFeatureExtractor import TransformerFeatureExtractor
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


def make_single_env(df, tech_indicator_list, seq_len):
    def _init():
        env = TradeEnv(df=df.copy(), tech_indicator_list=tech_indicator_list, lstm_seq_len=seq_len)
        env = FlattenObservation(env)  # ✅ 扁平化 obs，供 TransformerFeatureExtractor 使用
        return env
    return _init


def make_vec_env(df, tech_indicator_list, num_envs, seq_len):
    env_fns = [make_single_env(df, tech_indicator_list, seq_len) for _ in range(num_envs)]
    return SubprocVecEnv(env_fns)


class TradeAgent:
    def __init__(self):
        pass

    @staticmethod
    def get_model(model_kwargs=None, policy_kwargs=None, env=None, device="cpu"):
        model_kwargs = model_kwargs or {}
        policy_kwargs = policy_kwargs or {}

        # ✅ 默认设置 Transformer 作为特征提取器
        policy_kwargs['features_extractor_class'] = TransformerFeatureExtractor

        # ✅ 设置默认 Transformer 参数
        if 'features_extractor_kwargs' not in policy_kwargs or policy_kwargs['features_extractor_kwargs'] is None:
            policy_kwargs['features_extractor_kwargs'] = dict(
                seq_len=60,
                feature_dim=20,
                d_model=64,
                nhead=4,
                num_layers=2
            )

        model_kwargs['policy_kwargs'] = policy_kwargs
        model_kwargs['device'] = device
        model_kwargs['verbose'] = 1

        # 避免重复传入 gradient_steps
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

        # 序列长度，用于环境初始化和Transformer
        seq_len = 60
        if policy_kwargs is None:
            policy_kwargs = {}

        # 先创建环境
        env = make_vec_env(df, tech_indicator_list, num_envs, seq_len)

        # 自动根据环境 observation_space 计算 feature_dim
        obs_shape = env.observation_space.shape  # (obs_dim,)
        calculated_feature_dim = obs_shape[0] // seq_len
        print(f"环境 observation_space.shape: {obs_shape}，自动计算 feature_dim={calculated_feature_dim}")

        # 设置或更新 features_extractor_kwargs 中的 feature_dim 和 seq_len
        fe_kwargs = policy_kwargs.get('features_extractor_kwargs', {}) or {}
        fe_kwargs['seq_len'] = seq_len
        fe_kwargs['feature_dim'] = calculated_feature_dim
        policy_kwargs['features_extractor_kwargs'] = fe_kwargs

        print(f"\n=== 开始全量训练，共 {len(df)} 条数据 ===")

        model = None

        if model_load_path is not None:
            print(f"加载已有模型 {model_load_path} 进行增量训练")
            model = self.load_model(path=model_load_path, env=env, device=device, custom_objects=custom_objects)

        if model is None:
            print(f"模型加载失败或未提供，创建新模型")
            model = self.get_model(model_kwargs, policy_kwargs, env, device)

        model.set_env(env)

        total_timesteps = len(df) * single_step_num

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

        self.save_model(model_save_path, model)
        print(f"✅ 训练完成，模型保存至 {model_save_path}")
