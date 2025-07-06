import torch
import torch.nn as nn


class TransformerFeatureExtractor(nn.Module):
    def __init__(self, observation_space, seq_len, feature_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输入线性层，把 feature_dim 映射到 d_model
        self.input_linear = nn.Linear(feature_dim, d_model)

        # 输出特征维度，给SB3用
        self.features_dim = d_model

    def forward(self, observations):
        # observations形状通常是 (batch_size, seq_len * feature_dim) 或已flatten，需要恢复形状
        batch_size = observations.shape[0]
        x = observations.view(batch_size, self.seq_len, self.feature_dim)  # (B, T, F)

        x = self.input_linear(x)  # (B, T, d_model)
        x = self.transformer(x)  # (B, T, d_model)

        # 可以选用池化，比如取最后时间步特征，或者均值池化
        x = x.mean(dim=1)  # (B, d_model)

        return x
