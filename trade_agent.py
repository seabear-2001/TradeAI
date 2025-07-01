import os
import pandas as pd
import torch
from sb3_contrib import QRDQN

from trade_env import TradeEnv


class TradeAgent:
    def __init__(self):
        pass

    @staticmethod
    def get_model(model_kwargs=None, policy_kwargs=None, env=None, device="cpu"):
        return QRDQN(
            "MlpPolicy",
            env,
            verbose=1,
            device=device,
            **(model_kwargs or {}),
            policy_kwargs=policy_kwargs,
        )

    @staticmethod
    def save_model(path, model, last_train_ts):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': model.policy.state_dict(),
            'last_train_ts': last_train_ts,
            'model_kwargs': getattr(model, '_model_kwargs', None),
            'policy_kwargs': getattr(model, '_policy_kwargs', None),
        }, path)
        print(f"[模型已保存至 {path}]")

    def load_model(self, path, env=None, model_kwargs=None, policy_kwargs=None, device="cpu"):
        if not os.path.exists(path):
            print(f"模型文件不存在: {path}")
            return None, None
        model_data = torch.load(path, map_location=device)

        if model_kwargs is None:
            model_kwargs = model_data.get('model_kwargs', model_kwargs)
        if policy_kwargs is None:
            policy_kwargs = model_data.get('policy_kwargs', policy_kwargs)

        model = self.get_model(model_kwargs, policy_kwargs, env, device)
        model.policy.load_state_dict(model_data['model_state_dict'])
        model._model_kwargs = model_kwargs
        model._policy_kwargs = policy_kwargs
        last_train_ts = model_data.get('last_train_ts', 0)
        return model, last_train_ts

    def train_model(
        self,
        path,
        df,
        min_data_len=1000,
        single_step_num=3,
        num_envs=1,
        tech_indicator_list=None,
        model_kwargs=None,
        policy_kwargs=None,
        autoFeature=False,
        device="cpu",
    ):
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        check_timestamp_consistency(df)

        if os.path.exists(path):
            model_data = torch.load(path, map_location=device)
            old_model_kwargs = model_data.get("model_kwargs")
            old_policy_kwargs = model_data.get("policy_kwargs")
            last_train_ts = model_data.get("last_train_ts", 0)
        else:
            model_data = None
            old_model_kwargs = None
            old_policy_kwargs = None
            last_train_ts = 0

        full_train = not self._params_equal(model_kwargs, old_model_kwargs) or not self._params_equal(policy_kwargs, old_policy_kwargs)
        if full_train:
            data_to_train = df
            print(f"[模型 {path}] 超参数变化，执行全量训练（共 {len(data_to_train)} 条）")
        else:
            data_to_train = df[df['timestamp'].apply(lambda x: int(x.timestamp())) > last_train_ts]
            print(f"[模型 {path}] 增量训练（新增 {len(data_to_train)} 条）")
            if len(data_to_train) < min_data_len:
                print(f"[模型 {path}] 数据不足，跳过训练")
                return

        if autoFeature:
            print("🔧 特征工程中...")
            data_to_train = FeatureEngineer(tech_indicator_list).preprocess_data(data_to_train)

        env = make_vec_env(data_to_train, tech_indicator_list, num_envs)
        print(f"✅ {num_envs}环境并行构建完成")

        model, _ = self.load_model(path, env, model_kwargs, policy_kwargs, device)
        if model is None:
            model = self.get_model(model_kwargs, policy_kwargs, env, device)
            model._model_kwargs = model_kwargs
            model._policy_kwargs = policy_kwargs
        else:
            model.set_env(env)

        total_timesteps = len(data_to_train) * single_step_num * num_envs  # 乘以环境数，保持训练量
        model.learn(total_timesteps=total_timesteps, progress_bar=True)

        last_ts = int(data_to_train['timestamp'].max().timestamp())
        self.save_model(path, model, last_ts)
        print(f"[模型 {path}] 训练完成，最后时间戳更新为 {last_ts}")

    @staticmethod
    def _params_equal(params1, params2):
        if params1 is None:
            params1 = {}
        if params2 is None:
            params2 = {}
        return params1 == params2



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


from stable_baselines3.common.vec_env import SubprocVecEnv

def make_single_env(df, tech_indicator_list):
    def _init():
        return TradeEnv(df=df.copy(), tech_indicator_list=tech_indicator_list)
    return _init

def make_vec_env(df, tech_indicator_list, num_envs):
    env_fns = [make_single_env(df, tech_indicator_list) for _ in range(num_envs)]
    return SubprocVecEnv(env_fns)
