# ====================================================================== #
# МОДУЛЬ РЕАЛІЗАЦІЇ МОДЕЛІ НЕЙРОМЕРЕЖЕВОГО РОЗВ'ЯЗУВАЧА ОБЕРНЕНОЇ ЗАДАЧІ #
# ====================================================================== #
import torch
import torch.nn as nn
import math

# ==========================================
# 1. ДОДАЄ ЧАСОВИЙ КОНТЕКСТ (КООРДИНАТУ КРОКУ) ДЛЯ ТРАНСФОРМЕРА
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# ==========================================
# 2. ВЛАСНИЙ КЛАС ДЛЯ ПУЛІНГУ ПО ЧАСУ З МЕХАНІЗМОМ ATTENTION
# ==========================================
class TemporalAttentionPooling(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Аналізуємо кожен часовий крок (канал) і видаємо йому "вагу" важливості
        self.attn = nn.Sequential(
            nn.Conv1d(channels, channels // 2, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(channels // 2, 1, kernel_size=1),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # (Batch, Channels, Seq_len) -> (Batch, 128, 360)
        attn_weights = self.attn(x) # (Batch, 1, 360)
        # Множимо ознаки на їх вагу і сумуємо по часу
        pooled = torch.sum(x * attn_weights, dim=-1) # (Batch, 128)
        return pooled

# ==========================================
# 3. ГОЛОВНИЙ КЛАС СТРУКТУРИ НЕЙРОМЕРЕЖІ
# ==========================================
class Solver(nn.Module):
    def __init__(self, input_dim=8, output_dim=7):
        super().__init__()
        
        # Блок 1D-згорток: стиснення часової шкали 720 -> 360 кроків
        self.conv_extractor = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU()
        )
        
        # Блок уваги (Transformer Encoder)
        self.pos_encoder = PositionalEncoding(d_model=128)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, 
            nhead=4, 
            dim_feedforward=512, 
            dropout=0.1, 
            activation='gelu', 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # Роздільні пулінги для запобігання градієнтному голодуванню
        self.spatial_pool = nn.AdaptiveAvgPool1d(8)   # Жорсткий пулінг для геометрії
        self.dynamics_pool = nn.AdaptiveAvgPool1d(32) # М'який пулінг для фізики
        self.temporal_pool = TemporalAttentionPooling(channels=128) #Жорсткий пулінг для часу з увагою до піків
        
        # Повнозв'язна голова регресії простору (Spatial Head: x0, y0, z0)
        self.spatial_head = nn.Sequential(
            nn.Linear(128 * 8, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 3) 
        )

        # Повнозв'язна голова регресії фізики (Dynamics Head: D0, Q0, t0, beta)
        self.dynamics_head = nn.Sequential(
            nn.Linear(128 * 32, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 3) 
        )

        # Повнозв'язна голова регресії часу (Temporal Head: t0)
        self.temporal_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1) 
        )

    def forward(self, x):
        # Вхід x: (Batch, 720, 8) -> під Conv1d: (Batch, 8, 720)
        x = x.permute(0, 2, 1)
        x = self.conv_extractor(x) # (Batch, 128, 360)
        
        # Під Трансформер: (Batch, 360, 128)
        x = x.permute(0, 2, 1)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        
        # Повертаємо для пулінгу: (Batch, 128, 360)
        x = x.permute(0, 2, 1)
        
        # 1. Гілка геометрії (x0, y0, z0)
        x_spatial = self.spatial_pool(x)
        x_spatial = x_spatial.reshape(x_spatial.size(0), -1)
        out_spatial = self.spatial_head(x_spatial)
        
        # 2. Гілка фізики (D0, Q0, beta)
        x_dynamics = self.dynamics_pool(x)
        x_dynamics = x_dynamics.reshape(x_dynamics.size(0), -1)
        out_dynamics = self.dynamics_head(x_dynamics)
        
        # 3. Гілка часу (t0)
        x_temporal = self.temporal_pool(x)
        out_temporal = self.temporal_head(x_temporal)

        # 4. Збірка результату
        out = torch.cat([
            out_dynamics[:, :2], 
            out_spatial,
            out_temporal,
            out_dynamics[:, 2:],
        ], dim=1)
        
        return out
