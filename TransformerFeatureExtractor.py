import torch.nn as nn

class TransformerFeatureExtractor(nn.Module):
    def __init__(self, seq_len=60, feature_dim=24, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.d_model = d_model

        # 线性投影将输入特征维度映射到 d_model
        self.input_proj = nn.Linear(feature_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 后续根据需要输出特征维度（比如 d_model）
        self.output_dim = d_model

    def forward(self, observations):
        batch_size = observations.shape[0]  # 动态batch_size
        x = observations.view(batch_size, self.seq_len, self.feature_dim)  # (B, T, F)
        x = self.input_proj(x)  # (B, T, d_model)
        x = x.permute(1, 0, 2)  # Transformer需要 (T, B, d_model)
        x = self.transformer_encoder(x)  # (T, B, d_model)
        x = x.permute(1, 0, 2)  # (B, T, d_model)
        # 取序列最后一步的输出作为特征
        out = x[:, -1, :]  # (B, d_model)
        return out
