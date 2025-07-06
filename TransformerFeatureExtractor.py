import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TransformerFeatureExtractor(BaseFeaturesExtractor):
    """
    用于 QRDQN 的 Transformer 特征提取器。
    输入为展平的状态向量（seq_len * feature_dim），会 reshape 为 (batch, seq_len, feature_dim)。
    """

    def __init__(self, observation_space, seq_len=60, feature_dim=20, d_model=64, nhead=4, num_layers=2):
        # 最终输出是一个 d_model 向量，用于作为 QRDQN 的输入
        super().__init__(observation_space, features_dim=d_model)

        self.seq_len = seq_len
        self.feature_dim = feature_dim

        # 投影到 Transformer 输入维度
        self.input_proj = nn.Linear(feature_dim, d_model)

        # Transformer 编码器堆栈
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 池化：将 seq_len 降维为 1
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        :param observations: (batch_size, seq_len * feature_dim)
        :return: (batch_size, d_model)
        """
        batch_size = observations.shape[0]
        x = observations.view(batch_size, self.seq_len, self.feature_dim)  # 还原为 (B, T, F)

        x = self.input_proj(x)  # => (B, T, d_model)

        x = self.transformer_encoder(x)  # => (B, T, d_model)

        x = x.permute(0, 2, 1)  # => (B, d_model, T) for pooling

        x = self.pooling(x)  # => (B, d_model, 1)

        x = x.squeeze(-1)  # => (B, d_model)

        return x
