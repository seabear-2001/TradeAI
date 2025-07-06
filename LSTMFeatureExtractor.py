import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class LSTMFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, lstm_hidden_size=64):
        super().__init__(observation_space, features_dim=lstm_hidden_size)

        # 假设输入 shape 为 (seq_len, feature_dim)
        seq_len, feature_dim = observation_space.shape

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden_size,
            batch_first=True
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations shape: (batch_size, seq_len, feature_dim)
        lstm_out, (h_n, _) = self.lstm(observations)
        return h_n[-1]  # 返回最后一层的隐藏状态 (batch_size, hidden_size)
