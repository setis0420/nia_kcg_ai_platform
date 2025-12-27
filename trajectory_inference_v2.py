# -*- coding: utf-8 -*-
"""
선박 항적 예측 추론 V3
========================
V3: 속도 벡터 (Vx, Vy) 사용, MMSI 제거
- 4개 수치형 피처: lat, lon, vx, vy
- 3개 범주형 피처: start_area_id, end_area_id, grid_id
+ 해역 경계 마스크 적용 (육지 이탈 방지)
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# =============================================================================
# V3 모델 클래스 (속도 벡터 Vx, Vy 사용, MMSI 제거)
# =============================================================================

class LSTMTrajectoryModelV3(nn.Module):
    """V3 모델 (Embedding + LSTM) - 4 features, 3 categorical"""
    def __init__(self,
                 num_features=4,  # lat, lon, vx, vy
                 hidden_dim=128,
                 num_layers=2,
                 output_dim=4,
                 dropout=0.2,
                 num_start_area=50,
                 num_end_area=50,
                 num_grids=10000,
                 embed_dim=16):
        super().__init__()

        self.num_features = num_features
        self.embed_dim = embed_dim

        # V3: 3개 임베딩 (mmsi 제거)
        self.start_area_embed = nn.Embedding(num_start_area, embed_dim)
        self.end_area_embed = nn.Embedding(num_end_area, embed_dim)
        self.grid_embed = nn.Embedding(num_grids, embed_dim)

        lstm_input_dim = num_features + 3 * embed_dim  # 4 + 3*16 = 52

        self.lstm = nn.LSTM(lstm_input_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_num, x_cat):
        B, T, _ = x_num.shape

        # V3: x_cat shape = (B, T, 3) -> start, end, grid
        start_emb = self.start_area_embed(x_cat[:, :, 0])
        end_emb = self.end_area_embed(x_cat[:, :, 1])
        grid_emb = self.grid_embed(x_cat[:, :, 2])

        x = torch.cat([x_num, start_emb, end_emb, grid_emb], dim=-1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class LSTMAttentionModelV3(nn.Module):
    """V3 LSTM + Attention 모델"""
    def __init__(self,
                 num_features=4,
                 hidden_dim=128,
                 num_layers=2,
                 output_dim=4,
                 dropout=0.2,
                 num_start_area=50,
                 num_end_area=50,
                 num_grids=10000,
                 embed_dim=16,
                 n_heads=4):
        super().__init__()

        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        self.start_area_embed = nn.Embedding(num_start_area, embed_dim)
        self.end_area_embed = nn.Embedding(num_end_area, embed_dim)
        self.grid_embed = nn.Embedding(num_grids, embed_dim)

        lstm_input_dim = num_features + 3 * embed_dim
        self.lstm = nn.LSTM(lstm_input_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=dropout, bidirectional=False)

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x_num, x_cat):
        B, T, _ = x_num.shape

        start_emb = self.start_area_embed(x_cat[:, :, 0])
        end_emb = self.end_area_embed(x_cat[:, :, 1])
        grid_emb = self.grid_embed(x_cat[:, :, 2])

        x = torch.cat([x_num, start_emb, end_emb, grid_emb], dim=-1)
        lstm_out, _ = self.lstm(x)

        causal_mask = torch.triu(torch.ones(T, T, device=x_num.device), diagonal=1).bool()
        attn_out, _ = self.attention(
            lstm_out, lstm_out, lstm_out,
            attn_mask=causal_mask,
            need_weights=False
        )

        out = self.layer_norm(lstm_out + attn_out)
        return self.fc(out[:, -1, :])


# =============================================================================
# V2 모델 클래스 (하위 호환성)
# =============================================================================

class LSTMTrajectoryModelV2(nn.Module):
    """V2 모델 (Embedding + LSTM)"""
    def __init__(self,
                 num_features=5,
                 hidden_dim=128,
                 num_layers=2,
                 output_dim=5,
                 dropout=0.2,
                 num_mmsi=1000,
                 num_start_area=50,
                 num_end_area=50,
                 num_grids=10000,
                 embed_dim=16):
        super().__init__()

        self.num_features = num_features
        self.embed_dim = embed_dim

        self.mmsi_embed = nn.Embedding(num_mmsi, embed_dim)
        self.start_area_embed = nn.Embedding(num_start_area, embed_dim)
        self.end_area_embed = nn.Embedding(num_end_area, embed_dim)
        self.grid_embed = nn.Embedding(num_grids, embed_dim)

        lstm_input_dim = num_features + 4 * embed_dim

        self.lstm = nn.LSTM(lstm_input_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_num, x_cat):
        B, T, _ = x_num.shape

        mmsi_emb = self.mmsi_embed(x_cat[:, :, 0])
        start_emb = self.start_area_embed(x_cat[:, :, 1])
        end_emb = self.end_area_embed(x_cat[:, :, 2])
        grid_emb = self.grid_embed(x_cat[:, :, 3])

        x = torch.cat([x_num, mmsi_emb, start_emb, end_emb, grid_emb], dim=-1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class LSTMAttentionModel(nn.Module):
    """LSTM + Attention 모델"""
    def __init__(self,
                 num_features=5,
                 hidden_dim=128,
                 num_layers=2,
                 output_dim=5,
                 dropout=0.2,
                 num_mmsi=1000,
                 num_start_area=50,
                 num_end_area=50,
                 num_grids=10000,
                 embed_dim=16,
                 n_heads=4):
        super().__init__()

        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        self.mmsi_embed = nn.Embedding(num_mmsi, embed_dim)
        self.start_area_embed = nn.Embedding(num_start_area, embed_dim)
        self.end_area_embed = nn.Embedding(num_end_area, embed_dim)
        self.grid_embed = nn.Embedding(num_grids, embed_dim)

        lstm_input_dim = num_features + 4 * embed_dim
        self.lstm = nn.LSTM(lstm_input_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=dropout, bidirectional=False)

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x_num, x_cat):
        B, T, _ = x_num.shape

        mmsi_emb = self.mmsi_embed(x_cat[:, :, 0])
        start_emb = self.start_area_embed(x_cat[:, :, 1])
        end_emb = self.end_area_embed(x_cat[:, :, 2])
        grid_emb = self.grid_embed(x_cat[:, :, 3])

        x = torch.cat([x_num, mmsi_emb, start_emb, end_emb, grid_emb], dim=-1)
        lstm_out, _ = self.lstm(x)

        causal_mask = torch.triu(torch.ones(T, T, device=x_num.device), diagonal=1).bool()
        attn_out, _ = self.attention(
            lstm_out, lstm_out, lstm_out,
            attn_mask=causal_mask,
            need_weights=False
        )

        out = self.layer_norm(lstm_out + attn_out)
        return self.fc(out[:, -1, :])


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding"""
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# =============================================================================
# TFT Components
# =============================================================================

class GatedLinearUnit(nn.Module):
    """Gated Linear Unit (GLU)"""
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.dropout(self.sigmoid(self.gate(x)) * self.fc(x))


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, context_dim=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_dim = context_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        if context_dim is not None:
            self.context_fc = nn.Linear(context_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.elu = nn.ELU()
        self.glu = GatedLinearUnit(hidden_dim, output_dim, dropout)

        if input_dim != output_dim:
            self.skip_proj = nn.Linear(input_dim, output_dim)
        else:
            self.skip_proj = None
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x, context=None):
        residual = x if self.skip_proj is None else self.skip_proj(x)
        hidden = self.fc1(x)
        if self.context_dim is not None and context is not None:
            hidden = hidden + self.context_fc(context)
        hidden = self.elu(hidden)
        hidden = self.fc2(hidden)
        gated = self.glu(hidden)
        return self.layer_norm(residual + gated)


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network (VSN)"""
    def __init__(self, input_dim, num_inputs, hidden_dim, dropout=0.1, context_dim=None):
        super().__init__()
        self.num_inputs = num_inputs
        self.hidden_dim = hidden_dim

        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(input_dim, hidden_dim, hidden_dim, dropout)
            for _ in range(num_inputs)
        ])
        self.selection_grn = GatedResidualNetwork(
            input_dim * num_inputs, hidden_dim, num_inputs, dropout, context_dim
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs, context=None):
        processed = []
        for i, var_input in enumerate(inputs):
            processed.append(self.var_grns[i](var_input))

        concat_inputs = torch.cat(inputs, dim=-1)
        if context is not None and len(concat_inputs.shape) == 3:
            context = context.unsqueeze(1).expand(-1, concat_inputs.size(1), -1)

        selection_weights = self.selection_grn(concat_inputs, context)
        selection_weights = self.softmax(selection_weights)

        if len(selection_weights.shape) == 2:
            selection_weights = selection_weights.unsqueeze(-1)
            stacked = torch.stack(processed, dim=1)
            output = (stacked * selection_weights).sum(dim=1)
        else:
            selection_weights = selection_weights.unsqueeze(-1)
            stacked = torch.stack(processed, dim=2)
            output = (stacked * selection_weights).sum(dim=2)

        return output, selection_weights.squeeze(-1)


class InterpretableMultiHeadAttention(nn.Module):
    """Interpretable Multi-Head Attention"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_k ** -0.5

    def forward(self, query, key, value, mask=None):
        B, T, _ = query.shape
        Q = self.W_q(query).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)
        output = self.W_o(context)
        return output, attn_weights


class TemporalFusionTransformerV3(nn.Module):
    """V3 Temporal Fusion Transformer (4 features, 3 categorical)"""
    def __init__(self,
                 num_features=4,  # lat, lon, vx, vy
                 hidden_dim=128,
                 output_dim=4,
                 n_heads=4,
                 dropout=0.1,
                 num_start_area=50,
                 num_end_area=50,
                 num_grids=10000,
                 embed_dim=16,
                 num_lstm_layers=2):
        super().__init__()

        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.output_dim = output_dim

        # V3: 3개 임베딩 (mmsi 제거)
        self.start_area_embed = nn.Embedding(num_start_area, embed_dim)
        self.end_area_embed = nn.Embedding(num_end_area, embed_dim)
        self.grid_embed = nn.Embedding(num_grids, embed_dim)

        # Numeric projections
        self.numeric_projections = nn.ModuleList([
            nn.Linear(1, hidden_dim) for _ in range(num_features)
        ])
        self.cat_projections = nn.ModuleList([
            nn.Linear(embed_dim, hidden_dim) for _ in range(3)  # V3: 3개
        ])

        # Variable Selection
        total_vars = num_features + 3  # V3: 4 + 3 = 7
        self.var_selection = VariableSelectionNetwork(
            input_dim=hidden_dim,
            num_inputs=total_vars,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        # LSTM Encoder
        self.lstm_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        self.lstm_gate = GatedLinearUnit(hidden_dim, hidden_dim, dropout)
        self.lstm_norm = nn.LayerNorm(hidden_dim)

        # Static Enrichment
        self.static_enrichment = GatedResidualNetwork(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            dropout=dropout
        )

        # Temporal Self-Attention
        self.multihead_attn = InterpretableMultiHeadAttention(
            d_model=hidden_dim,
            n_heads=n_heads,
            dropout=dropout
        )
        self.attn_gate = GatedLinearUnit(hidden_dim, hidden_dim, dropout)
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # Feed-Forward
        self.ff_grn = GatedResidualNetwork(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim * 4,
            output_dim=hidden_dim,
            dropout=dropout
        )

        # Output Layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x_num, x_cat):
        B, T, _ = x_num.shape

        # Embedding
        numeric_vars = []
        for i in range(self.num_features):
            var = x_num[:, :, i:i+1]
            projected = self.numeric_projections[i](var)
            numeric_vars.append(projected)

        # V3: 3개 범주형 (start, end, grid)
        start_emb = self.cat_projections[0](self.start_area_embed(x_cat[:, :, 0]))
        end_emb = self.cat_projections[1](self.end_area_embed(x_cat[:, :, 1]))
        grid_emb = self.cat_projections[2](self.grid_embed(x_cat[:, :, 2]))
        cat_vars = [start_emb, end_emb, grid_emb]

        # Variable Selection
        all_vars = numeric_vars + cat_vars
        selected, var_weights = self.var_selection(all_vars)

        # LSTM Encoder
        lstm_out, _ = self.lstm_encoder(selected)
        lstm_gated = self.lstm_gate(lstm_out)
        lstm_out = self.lstm_norm(selected + lstm_gated)

        # Static Enrichment
        enriched = self.static_enrichment(lstm_out)

        # Temporal Self-Attention
        mask = torch.triu(torch.ones(T, T, device=x_num.device), diagonal=1).bool()
        mask = ~mask
        attn_out, attn_weights = self.multihead_attn(enriched, enriched, enriched, mask)
        attn_gated = self.attn_gate(attn_out)
        attn_out = self.attn_norm(enriched + attn_gated)

        # Feed-Forward
        ff_out = self.ff_grn(attn_out)

        # Output
        final_hidden = ff_out[:, -1, :]
        return self.output_layer(final_hidden)


class TemporalFusionTransformer(nn.Module):
    """Temporal Fusion Transformer (TFT) - V2 호환"""
    def __init__(self,
                 num_features=5,
                 hidden_dim=128,
                 output_dim=5,
                 n_heads=4,
                 dropout=0.1,
                 num_mmsi=1000,
                 num_start_area=50,
                 num_end_area=50,
                 num_grids=10000,
                 embed_dim=16,
                 num_lstm_layers=2):
        super().__init__()

        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.output_dim = output_dim

        # Embeddings
        self.mmsi_embed = nn.Embedding(num_mmsi, embed_dim)
        self.start_area_embed = nn.Embedding(num_start_area, embed_dim)
        self.end_area_embed = nn.Embedding(num_end_area, embed_dim)
        self.grid_embed = nn.Embedding(num_grids, embed_dim)

        # Numeric projections
        self.numeric_projections = nn.ModuleList([
            nn.Linear(1, hidden_dim) for _ in range(num_features)
        ])
        self.cat_projections = nn.ModuleList([
            nn.Linear(embed_dim, hidden_dim) for _ in range(4)
        ])

        # Variable Selection
        total_vars = num_features + 4
        self.var_selection = VariableSelectionNetwork(
            input_dim=hidden_dim,
            num_inputs=total_vars,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        # LSTM Encoder
        self.lstm_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        self.lstm_gate = GatedLinearUnit(hidden_dim, hidden_dim, dropout)
        self.lstm_norm = nn.LayerNorm(hidden_dim)

        # Static Enrichment
        self.static_enrichment = GatedResidualNetwork(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            dropout=dropout
        )

        # Temporal Self-Attention
        self.multihead_attn = InterpretableMultiHeadAttention(
            d_model=hidden_dim,
            n_heads=n_heads,
            dropout=dropout
        )
        self.attn_gate = GatedLinearUnit(hidden_dim, hidden_dim, dropout)
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # Feed-Forward
        self.ff_grn = GatedResidualNetwork(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim * 4,
            output_dim=hidden_dim,
            dropout=dropout
        )

        # Output Layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x_num, x_cat):
        B, T, _ = x_num.shape

        # Embedding
        numeric_vars = []
        for i in range(self.num_features):
            var = x_num[:, :, i:i+1]
            projected = self.numeric_projections[i](var)
            numeric_vars.append(projected)

        mmsi_emb = self.cat_projections[0](self.mmsi_embed(x_cat[:, :, 0]))
        start_emb = self.cat_projections[1](self.start_area_embed(x_cat[:, :, 1]))
        end_emb = self.cat_projections[2](self.end_area_embed(x_cat[:, :, 2]))
        grid_emb = self.cat_projections[3](self.grid_embed(x_cat[:, :, 3]))
        cat_vars = [mmsi_emb, start_emb, end_emb, grid_emb]

        # Variable Selection
        all_vars = numeric_vars + cat_vars
        selected, var_weights = self.var_selection(all_vars)

        # LSTM Encoder
        lstm_out, _ = self.lstm_encoder(selected)
        lstm_gated = self.lstm_gate(lstm_out)
        lstm_out = self.lstm_norm(selected + lstm_gated)

        # Static Enrichment
        enriched = self.static_enrichment(lstm_out)

        # Temporal Self-Attention
        mask = torch.triu(torch.ones(T, T, device=x_num.device), diagonal=1).bool()
        mask = ~mask
        attn_out, attn_weights = self.multihead_attn(enriched, enriched, enriched, mask)
        attn_gated = self.attn_gate(attn_out)
        attn_out = self.attn_norm(enriched + attn_gated)

        # Feed-Forward
        ff_out = self.ff_grn(attn_out)

        # Output
        final_hidden = ff_out[:, -1, :]
        return self.output_layer(final_hidden)


def step_latlon_dead_reckoning(lat, lon, sog_kn, cog_deg, dt_minutes=1):
    """Dead Reckoning으로 다음 위치 계산"""
    v = float(sog_kn) * 0.514444
    d = v * float(dt_minutes) * 60.0
    brng = np.radians(float(cog_deg))

    R = 6371000.0
    lat1 = np.radians(float(lat))
    lon1 = np.radians(float(lon))

    lat2 = np.arcsin(np.sin(lat1) * np.cos(d / R) +
                     np.cos(lat1) * np.sin(d / R) * np.cos(brng))
    lon2 = lon1 + np.arctan2(np.sin(brng) * np.sin(d / R) * np.cos(lat1),
                             np.cos(d / R) - np.sin(lat1) * np.sin(lat2))

    lat2 = np.degrees(lat2)
    lon2 = np.degrees(lon2)
    lon2 = (lon2 + 540) % 360 - 180
    return lat2, lon2


def compute_grid_id(lat, lon, lat_min, lon_min, grid_size, num_cols):
    """위경도를 격자 ID로 변환"""
    grid_row = int((lat - lat_min) / grid_size)
    grid_col = int((lon - lon_min) / grid_size)
    return grid_row * num_cols + grid_col


def load_sea_boundary(boundary_path):
    """해역 경계 데이터 로드"""
    if not os.path.exists(boundary_path):
        return None

    data = np.load(boundary_path, allow_pickle=True)

    sea_mask = {
        'valid_mask': data['valid_mask'],
        'lat_min': float(data['mask_lat_min']),
        'lat_max': float(data['mask_lat_max']),
        'lon_min': float(data['mask_lon_min']),
        'lon_max': float(data['mask_lon_max']),
        'grid_size': float(data['mask_grid_size']),
        'num_rows': int(data['mask_num_rows']),
        'num_cols': int(data['mask_num_cols']),
    }

    route_info = {
        'density_grid': data['density_grid'],
        'high_density_mask': data['high_density_mask'],
        'lat_min': float(data['route_lat_min']),
        'lat_max': float(data['route_lat_max']),
        'lon_min': float(data['route_lon_min']),
        'lon_max': float(data['route_lon_max']),
        'grid_size': float(data['route_grid_size']),
    }

    return {'sea_mask': sea_mask, 'routes': route_info}


def is_valid_sea_position(lat, lon, sea_mask):
    """위치가 유효 해역인지 확인"""
    if sea_mask is None:
        return True

    row = int((lat - sea_mask['lat_min']) / sea_mask['grid_size'])
    col = int((lon - sea_mask['lon_min']) / sea_mask['grid_size'])

    if 0 <= row < sea_mask['num_rows'] and 0 <= col < sea_mask['num_cols']:
        return bool(sea_mask['valid_mask'][row, col])
    return False


def find_nearest_valid_position(lat, lon, prev_lat, prev_lon, sea_mask, max_search_radius=20):
    """
    가장 가까운 유효 해역 위치 찾기

    전략:
    1. 현재 위치가 유효하면 그대로 반환
    2. 유효하지 않으면 이전 위치와 현재 위치 사이에서 유효한 지점 찾기
    3. 그래도 없으면 나선형으로 주변 검색
    4. 최후의 수단으로 이전 위치 반환
    """
    if sea_mask is None:
        return lat, lon

    # 이미 유효하면 그대로
    if is_valid_sea_position(lat, lon, sea_mask):
        return lat, lon

    grid_size = sea_mask['grid_size']

    # 전략 1: 이전 위치와 현재 위치 사이에서 유효한 지점 찾기 (이진 탐색)
    for ratio in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
        mid_lat = prev_lat + (lat - prev_lat) * ratio
        mid_lon = prev_lon + (lon - prev_lon) * ratio
        if is_valid_sea_position(mid_lat, mid_lon, sea_mask):
            return mid_lat, mid_lon

    # 전략 2: 나선형으로 주변 검색
    row = int((lat - sea_mask['lat_min']) / grid_size)
    col = int((lon - sea_mask['lon_min']) / grid_size)

    for radius in range(1, max_search_radius + 1):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if abs(dr) != radius and abs(dc) != radius:
                    continue

                nr, nc = row + dr, col + dc
                if 0 <= nr < sea_mask['num_rows'] and 0 <= nc < sea_mask['num_cols']:
                    if sea_mask['valid_mask'][nr, nc]:
                        new_lat = sea_mask['lat_min'] + (nr + 0.5) * grid_size
                        new_lon = sea_mask['lon_min'] + (nc + 0.5) * grid_size
                        return new_lat, new_lon

    # 전략 3: 이전 위치가 유효하면 이전 위치 반환
    if is_valid_sea_position(prev_lat, prev_lon, sea_mask):
        return prev_lat, prev_lon

    # 최후: 원래 위치 그대로 (경고만)
    return lat, lon


class TrajectoryInferenceV2:
    """V2 추론 클래스 (Categorical + Grid 지원 + 해역 경계)"""

    def __init__(self, model_path, scaler_path, seq_len=None, device=None, boundary_path=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # 해역 경계 로드 (자동 탐색)
        if boundary_path is None:
            # scaler와 같은 폴더 또는 prepared_data 폴더에서 탐색
            scaler_dir = os.path.dirname(scaler_path)
            candidates = [
                os.path.join(scaler_dir, "sea_boundary.npz"),
                os.path.join(scaler_dir, "..", "sea_boundary.npz"),
                "prepared_data/sea_boundary.npz",
                "sea_boundary.npz",
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    boundary_path = cand
                    break

        self.boundary_data = None
        self.sea_mask = None
        if boundary_path and os.path.exists(boundary_path):
            self.boundary_data = load_sea_boundary(boundary_path)
            if self.boundary_data:
                self.sea_mask = self.boundary_data['sea_mask']
                print(f"[InferenceV2] 해역 경계 로드: {boundary_path}")
                valid_count = np.sum(self.sea_mask['valid_mask'])
                print(f"[InferenceV2] 유효 해역 격자: {valid_count:,} 개")

        ckpt = np.load(scaler_path, allow_pickle=True)

        # 정규화 파라미터
        self.x_mean = ckpt["x_mean"].astype(np.float32)
        self.x_std  = ckpt["x_std"].astype(np.float32)
        self.y_mean = ckpt["y_mean"].astype(np.float32)
        self.y_std  = ckpt["y_std"].astype(np.float32)

        # 피처 컬럼
        self.numeric_cols = [str(x) for x in ckpt["numeric_cols"]]
        self.target_cols  = [str(x) for x in ckpt["target_cols"]]

        # 시퀀스 정보
        if seq_len is not None:
            self.seq_len = int(seq_len)
        elif "seq_len" in ckpt:
            self.seq_len = int(ckpt["seq_len"])
        else:
            self.seq_len = 80

        # Categorical vocab 로드
        self.mmsi_vocab = eval(str(ckpt["mmsi_vocab"]))
        self.start_vocab = eval(str(ckpt["start_vocab"]))
        self.end_vocab = eval(str(ckpt["end_vocab"]))

        num_mmsi = int(ckpt["num_mmsi"])
        num_start_area = int(ckpt["num_start_area"])
        num_end_area = int(ckpt["num_end_area"])

        # 격자 정보
        self.grid_lat_min = float(ckpt["grid_info_lat_min"])
        self.grid_lon_min = float(ckpt["grid_info_lon_min"])
        self.grid_size = float(ckpt["grid_size"])
        self.num_rows = int(ckpt["num_rows"])
        self.num_cols = int(ckpt["num_cols"])
        self.total_grids = int(ckpt["total_grids"])

        # COG 미러링 여부 (훈련 시 sin_cog를 음수로 변환했는지)
        self.cog_mirror = bool(ckpt["cog_mirror"]) if "cog_mirror" in ckpt else False
        if self.cog_mirror:
            print(f"[InferenceV2] COG 미러링 적용됨 (sin_cog 부호 반전)")

        # Heading Inertia: COG 관성 강도 (높을수록 이전 방향 유지)
        self.heading_inertia_lambda = float(ckpt["heading_inertia_lambda"]) if "heading_inertia_lambda" in ckpt else 0.0
        if self.heading_inertia_lambda > 0:
            print(f"[InferenceV2] Heading Inertia: {self.heading_inertia_lambda}")

        embed_dim = int(ckpt["embed_dim"]) if "embed_dim" in ckpt else 16

        # 항로 범위
        if "lat_bounds" in ckpt and "lon_bounds" in ckpt:
            self.lat_bounds = tuple(ckpt["lat_bounds"])
            self.lon_bounds = tuple(ckpt["lon_bounds"])
            print(f"[InferenceV2] 항로 범위: lat={self.lat_bounds[0]:.4f}~{self.lat_bounds[1]:.4f}, lon={self.lon_bounds[0]:.4f}~{self.lon_bounds[1]:.4f}")
        else:
            self.lat_bounds = None
            self.lon_bounds = None

        print(f"[InferenceV2] seq_len={self.seq_len}")
        print(f"[InferenceV2] 격자: {self.num_rows}x{self.num_cols} = {self.total_grids} (크기: {self.grid_size}도)")
        print(f"[InferenceV2] Categorical: mmsi={num_mmsi}, start_area={num_start_area}, end_area={num_end_area}")

        # 모델 타입 자동 감지 (state_dict 키 분석)
        state = torch.load(model_path, map_location=self.device, weights_only=False)

        # 모델 타입 판별
        state_keys = set(state.keys())

        # TFT 모델: var_selection, lstm_encoder 등의 키가 있음
        if "var_selection.var_grns.0.fc1.weight" in state_keys or "lstm_encoder.weight_ih_l0" in state_keys:
            model_type = "tft"
        # LSTM+Attention: attention.in_proj_weight 키가 있음
        elif "attention.in_proj_weight" in state_keys:
            model_type = "lstm_attn"
        else:
            model_type = "lstm"

        print(f"[InferenceV2] 감지된 모델 타입: {model_type}")

        # 모델 생성
        if model_type == "tft":
            self.model = TemporalFusionTransformer(
                num_features=len(self.numeric_cols),
                output_dim=len(self.target_cols),
                num_mmsi=num_mmsi,
                num_start_area=num_start_area,
                num_end_area=num_end_area,
                num_grids=self.total_grids + 1,
                embed_dim=embed_dim,
            ).to(self.device)
        elif model_type == "lstm_attn":
            self.model = LSTMAttentionModel(
                num_features=len(self.numeric_cols),
                output_dim=len(self.target_cols),
                num_mmsi=num_mmsi,
                num_start_area=num_start_area,
                num_end_area=num_end_area,
                num_grids=self.total_grids + 1,
                embed_dim=embed_dim,
            ).to(self.device)
        else:
            self.model = LSTMTrajectoryModelV2(
                num_features=len(self.numeric_cols),
                output_dim=len(self.target_cols),
                num_mmsi=num_mmsi,
                num_start_area=num_start_area,
                num_end_area=num_end_area,
                num_grids=self.total_grids + 1,
                embed_dim=embed_dim,
            ).to(self.device)

        self.model.load_state_dict(state)
        self.model.eval()

    def _get_mmsi_id(self, mmsi):
        """MMSI를 ID로 변환"""
        return self.mmsi_vocab.get(mmsi, 0)

    def _get_start_area_id(self, area):
        """start_area를 ID로 변환"""
        return self.start_vocab.get(area, 0)

    def _get_end_area_id(self, area):
        """end_area를 ID로 변환"""
        return self.end_vocab.get(area, 0)

    def _get_grid_id(self, lat, lon):
        """위경도를 격자 ID로 변환"""
        grid_id = compute_grid_id(lat, lon, self.grid_lat_min, self.grid_lon_min,
                                   self.grid_size, self.num_cols)
        return np.clip(grid_id, 0, self.total_grids)

    def _prepare_hist(self, df, mmsi=None, start_area=None, end_area=None):
        """입력 데이터 준비"""
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        for c in ["lat","lon","sog","cog"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["datetime","lat","lon","sog","cog"]).sort_values("datetime").reset_index(drop=True)

        if len(df) < self.seq_len:
            raise ValueError(f"rows({len(df)}) < seq_len({self.seq_len})")

        hist = df.iloc[-self.seq_len:].copy()
        sin_cog = np.sin(np.radians(hist["cog"]))
        cos_cog = np.cos(np.radians(hist["cog"]))

        # 훈련 시 적용된 COG 미러링 적용
        if self.cog_mirror:
            sin_cog = -sin_cog

        hist["sin_cog"] = sin_cog
        hist["cos_cog"] = cos_cog

        # Categorical ID
        if mmsi is None:
            mmsi = hist["mmsi"].iloc[0] if "mmsi" in hist.columns else 0
        if start_area is None:
            start_area = hist["start_area"].iloc[0] if "start_area" in hist.columns else "unknown"
        if end_area is None:
            end_area = hist["end_area"].iloc[0] if "end_area" in hist.columns else "unknown"

        self._current_mmsi_id = self._get_mmsi_id(mmsi)
        self._current_start_id = self._get_start_area_id(start_area)
        self._current_end_id = self._get_end_area_id(end_area)

        return hist

    def _clamp_to_bounds(self, lat, lon, prev_lat, prev_lon):
        """항로 범위 벗어나면 보정"""
        if self.lat_bounds is None or self.lon_bounds is None:
            return lat, lon, False

        out_of_bounds = False
        clamped_lat = lat
        clamped_lon = lon

        if lat < self.lat_bounds[0]:
            clamped_lat = self.lat_bounds[0]
            out_of_bounds = True
        elif lat > self.lat_bounds[1]:
            clamped_lat = self.lat_bounds[1]
            out_of_bounds = True

        if lon < self.lon_bounds[0]:
            clamped_lon = self.lon_bounds[0]
            out_of_bounds = True
        elif lon > self.lon_bounds[1]:
            clamped_lon = self.lon_bounds[1]
            out_of_bounds = True

        if out_of_bounds:
            clamped_lat = (prev_lat + clamped_lat) / 2
            clamped_lon = (prev_lon + clamped_lon) / 2

        return clamped_lat, clamped_lon, out_of_bounds

    def predict_multi_from_df(self, df, n_steps=80,
                              mmsi=None, start_area=None, end_area=None,
                              sog_clip=(0.0, 35.0),
                              sog_min_ratio=0.7,
                              use_model_latlon=False,
                              enforce_bounds=True,
                              enforce_sea_boundary=True,
                              heading_inertia=None):
        """
        다중 스텝 예측 (1분 간격)

        Parameters:
        -----------
        df: 입력 DataFrame (datetime, lat, lon, sog, cog 필요, 1분 간격 보간 데이터)
        n_steps: 예측 스텝 수
        mmsi, start_area, end_area: Categorical 값 (None이면 df에서 추출)
        sog_clip: SOG 클리핑 범위 (min, max)
        sog_min_ratio: 입력 데이터 평균 SOG 대비 최소 비율 (0.7 = 70% 이상 유지)
        enforce_sea_boundary: 해역 경계 적용 여부 (True: 육지 이탈 시 보정)
        heading_inertia: COG 관성 가중치 (0~1, 높을수록 이전 COG 유지, None이면 scaler 값 사용)
        """
        # heading_inertia 결정: 파라미터 > scaler 저장값 > 기본값 0
        if heading_inertia is None:
            # heading_inertia_lambda를 0~1 범위로 변환 (0.1 -> 약 0.1 가중치)
            heading_inertia = min(self.heading_inertia_lambda, 0.5)

        hist = self._prepare_hist(df, mmsi, start_area, end_area)

        # 수치형 피처
        X_num = hist[["lat","lon","sog","sin_cog","cos_cog"]].values.astype(np.float32)
        Xn_num = (X_num - self.x_mean) / self.x_std

        # 입력 데이터의 평균 SOG를 기준으로 최소값 설정
        input_sog_mean = float(hist["sog"].mean())
        sog_min_threshold = input_sog_mean * sog_min_ratio

        # Categorical 피처 (시퀀스 전체에 동일 값)
        X_cat = np.zeros((self.seq_len, 4), dtype=np.int64)
        X_cat[:, 0] = self._current_mmsi_id
        X_cat[:, 1] = self._current_start_id
        X_cat[:, 2] = self._current_end_id

        # 격자 ID는 각 위치마다 다름
        for i in range(self.seq_len):
            X_cat[i, 3] = self._get_grid_id(hist["lat"].iloc[i], hist["lon"].iloc[i])

        cur_lat = float(hist["lat"].iloc[-1])
        cur_lon = float(hist["lon"].iloc[-1])
        last_time = hist["datetime"].iloc[-1]

        # 이전 COG 추적 (관성 적용용)
        prev_cog = float(hist["cog"].iloc[-1])

        preds = []
        bound_violations = 0
        sea_boundary_corrections = 0

        for _ in range(int(n_steps)):
            x_num_t = torch.from_numpy(Xn_num).unsqueeze(0).to(self.device)
            x_cat_t = torch.from_numpy(X_cat).unsqueeze(0).to(self.device)

            with torch.inference_mode():
                y_hat_n = self.model(x_num_t, x_cat_t).squeeze(0).cpu().numpy().astype(np.float32)

            y_hat = y_hat_n * self.y_std.squeeze(0) + self.y_mean.squeeze(0)
            pred_lat_m, pred_lon_m, pred_sog, pred_sin, pred_cos = y_hat.tolist()

            # COG 미러링 역변환 (훈련 시 sin_cog가 반전되었으면 되돌림)
            if self.cog_mirror:
                pred_sin = -pred_sin

            model_cog = np.degrees(np.arctan2(pred_sin, pred_cos))
            model_cog = (model_cog + 360) % 360

            # Heading Inertia 적용: 이전 COG와 모델 예측 COG를 블렌딩
            if heading_inertia > 0:
                # 각도 차이를 -180 ~ 180 범위로 정규화
                cog_diff = model_cog - prev_cog
                if cog_diff > 180:
                    cog_diff -= 360
                elif cog_diff < -180:
                    cog_diff += 360

                # 관성 적용: 변화량을 줄임
                adjusted_diff = cog_diff * (1 - heading_inertia)
                pred_cog = prev_cog + adjusted_diff
                pred_cog = (pred_cog + 360) % 360
            else:
                pred_cog = model_cog

            # SOG 클리핑: 최소값은 입력 데이터 평균의 일정 비율 이상 유지
            effective_sog_min = max(sog_clip[0], sog_min_threshold)
            pred_sog = float(np.clip(pred_sog, effective_sog_min, sog_clip[1]))

            if use_model_latlon:
                next_lat, next_lon = float(pred_lat_m), float(pred_lon_m)
            else:
                next_lat, next_lon = step_latlon_dead_reckoning(cur_lat, cur_lon, pred_sog, pred_cog, dt_minutes=1)

            if enforce_bounds:
                next_lat, next_lon, violated = self._clamp_to_bounds(next_lat, next_lon, cur_lat, cur_lon)
                if violated:
                    bound_violations += 1

            # 해역 경계 적용: 육지로 예측되면 가장 가까운 유효 해역으로 보정
            if enforce_sea_boundary and self.sea_mask is not None:
                if not is_valid_sea_position(next_lat, next_lon, self.sea_mask):
                    corrected_lat, corrected_lon = find_nearest_valid_position(
                        next_lat, next_lon, cur_lat, cur_lon, self.sea_mask
                    )
                    next_lat, next_lon = corrected_lat, corrected_lon
                    sea_boundary_corrections += 1

            last_time = last_time + pd.Timedelta(minutes=1)
            preds.append([last_time, next_lat, next_lon, pred_sog, pred_cog])

            cur_lat, cur_lon = next_lat, next_lon
            prev_cog = pred_cog  # 다음 스텝의 관성 계산용

            # 시퀀스 업데이트
            new_sin_cog = np.sin(np.radians(pred_cog))
            new_cos_cog = np.cos(np.radians(pred_cog))

            # COG 미러링 적용 (다음 입력에도 동일하게)
            if self.cog_mirror:
                new_sin_cog = -new_sin_cog

            new_row_num = np.array([
                cur_lat, cur_lon, pred_sog,
                new_sin_cog,
                new_cos_cog,
            ], dtype=np.float32)

            new_row_num_n = (new_row_num - self.x_mean.squeeze(0)) / self.x_std.squeeze(0)
            Xn_num = np.vstack([Xn_num[1:], new_row_num_n])

            # Categorical 업데이트 (격자 ID만 변경)
            new_grid_id = self._get_grid_id(cur_lat, cur_lon)
            new_cat_row = np.array([self._current_mmsi_id, self._current_start_id,
                                     self._current_end_id, new_grid_id], dtype=np.int64)
            X_cat = np.vstack([X_cat[1:], new_cat_row])

        if bound_violations > 0:
            print(f"[InferenceV2] 경고: {bound_violations}회 항로 범위 이탈 보정됨")
        if sea_boundary_corrections > 0:
            print(f"[InferenceV2] 경고: {sea_boundary_corrections}회 육지 이탈 보정됨 (해역 경계 적용)")

        return pd.DataFrame(preds, columns=["datetime","pred_lat","pred_lon","pred_sog","pred_cog"])


class TrajectoryInferenceV3:
    """
    V3 추론 클래스 (속도 벡터 Vx, Vy 사용, MMSI 제거)

    - 4개 수치형 피처: lat, lon, vx, vy
    - 3개 범주형 피처: start_area_id, end_area_id, grid_id
    """

    def __init__(self, model_path, scaler_path, seq_len=None, device=None, boundary_path=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # 해역 경계 로드 (자동 탐색)
        if boundary_path is None:
            scaler_dir = os.path.dirname(scaler_path)
            candidates = [
                os.path.join(scaler_dir, "sea_boundary.npz"),
                os.path.join(scaler_dir, "..", "sea_boundary.npz"),
                "prepared_data/sea_boundary.npz",
                "sea_boundary.npz",
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    boundary_path = cand
                    break

        self.boundary_data = None
        self.sea_mask = None
        if boundary_path and os.path.exists(boundary_path):
            self.boundary_data = load_sea_boundary(boundary_path)
            if self.boundary_data:
                self.sea_mask = self.boundary_data['sea_mask']
                print(f"[InferenceV3] 해역 경계 로드: {boundary_path}")
                valid_count = np.sum(self.sea_mask['valid_mask'])
                print(f"[InferenceV3] 유효 해역 격자: {valid_count:,} 개")

        ckpt = np.load(scaler_path, allow_pickle=True)

        # V3 버전 확인
        self.version = str(ckpt.get("version", "v2"))
        if self.version != "v3":
            print(f"[경고] scaler 버전이 {self.version}입니다. V3 모델이 아닐 수 있습니다.")

        # 정규화 파라미터
        self.x_mean = ckpt["x_mean"].astype(np.float32)
        self.x_std  = ckpt["x_std"].astype(np.float32)
        self.y_mean = ckpt["y_mean"].astype(np.float32)
        self.y_std  = ckpt["y_std"].astype(np.float32)

        # 피처 컬럼
        self.numeric_cols = [str(x) for x in ckpt["numeric_cols"]]  # lat, lon, vx, vy
        self.target_cols  = [str(x) for x in ckpt["target_cols"]]

        # 시퀀스 정보
        if seq_len is not None:
            self.seq_len = int(seq_len)
        elif "seq_len" in ckpt:
            self.seq_len = int(ckpt["seq_len"])
        else:
            self.seq_len = 80

        # V3: MMSI vocab 없음
        self.start_vocab = eval(str(ckpt["start_vocab"]))
        self.end_vocab = eval(str(ckpt["end_vocab"]))

        num_start_area = int(ckpt["num_start_area"])
        num_end_area = int(ckpt["num_end_area"])

        # 격자 정보
        self.grid_lat_min = float(ckpt["grid_info_lat_min"])
        self.grid_lon_min = float(ckpt["grid_info_lon_min"])
        self.grid_size = float(ckpt["grid_size"])
        self.num_rows = int(ckpt["num_rows"])
        self.num_cols = int(ckpt["num_cols"])
        self.total_grids = int(ckpt["total_grids"])

        embed_dim = int(ckpt["embed_dim"]) if "embed_dim" in ckpt else 16

        # 항로 범위
        if "lat_bounds" in ckpt and "lon_bounds" in ckpt:
            self.lat_bounds = tuple(ckpt["lat_bounds"])
            self.lon_bounds = tuple(ckpt["lon_bounds"])
            print(f"[InferenceV3] 항로 범위: lat={self.lat_bounds[0]:.4f}~{self.lat_bounds[1]:.4f}, lon={self.lon_bounds[0]:.4f}~{self.lon_bounds[1]:.4f}")
        else:
            self.lat_bounds = None
            self.lon_bounds = None

        print(f"[InferenceV3] V3 모델 (속도 벡터 Vx, Vy)")
        print(f"[InferenceV3] seq_len={self.seq_len}")
        print(f"[InferenceV3] 격자: {self.num_rows}x{self.num_cols} = {self.total_grids} (크기: {self.grid_size}도)")
        print(f"[InferenceV3] Categorical: start_area={num_start_area}, end_area={num_end_area}")

        # 모델 타입 자동 감지
        state = torch.load(model_path, map_location=self.device, weights_only=False)
        state_keys = set(state.keys())

        # V3 모델: mmsi_embed 키가 없음
        if "mmsi_embed.weight" in state_keys:
            raise ValueError("V2 모델입니다. TrajectoryInferenceV2를 사용하세요.")

        # 모델 타입 판별
        if "var_selection.var_grns.0.fc1.weight" in state_keys or "lstm_encoder.weight_ih_l0" in state_keys:
            model_type = "tft"
        elif "attention.in_proj_weight" in state_keys:
            model_type = "lstm_attn"
        else:
            model_type = "lstm"

        print(f"[InferenceV3] 감지된 모델 타입: {model_type}")

        # V3 모델 생성
        if model_type == "tft":
            self.model = TemporalFusionTransformerV3(
                num_features=len(self.numeric_cols),
                output_dim=len(self.target_cols),
                num_start_area=num_start_area,
                num_end_area=num_end_area,
                num_grids=self.total_grids + 1,
                embed_dim=embed_dim,
            ).to(self.device)
        elif model_type == "lstm_attn":
            self.model = LSTMAttentionModelV3(
                num_features=len(self.numeric_cols),
                output_dim=len(self.target_cols),
                num_start_area=num_start_area,
                num_end_area=num_end_area,
                num_grids=self.total_grids + 1,
                embed_dim=embed_dim,
            ).to(self.device)
        else:
            self.model = LSTMTrajectoryModelV3(
                num_features=len(self.numeric_cols),
                output_dim=len(self.target_cols),
                num_start_area=num_start_area,
                num_end_area=num_end_area,
                num_grids=self.total_grids + 1,
                embed_dim=embed_dim,
            ).to(self.device)

        self.model.load_state_dict(state)
        self.model.eval()

    def _get_start_area_id(self, area):
        """start_area를 ID로 변환"""
        return self.start_vocab.get(area, 0)

    def _get_end_area_id(self, area):
        """end_area를 ID로 변환"""
        return self.end_vocab.get(area, 0)

    def _get_grid_id(self, lat, lon):
        """위경도를 격자 ID로 변환"""
        grid_id = compute_grid_id(lat, lon, self.grid_lat_min, self.grid_lon_min,
                                   self.grid_size, self.num_cols)
        return np.clip(grid_id, 0, self.total_grids)

    def _prepare_hist(self, df, start_area=None, end_area=None):
        """입력 데이터 준비 (V3: Vx, Vy 계산)"""
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        for c in ["lat", "lon", "sog", "cog"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["datetime", "lat", "lon", "sog", "cog"]).sort_values("datetime").reset_index(drop=True)

        if len(df) < self.seq_len:
            raise ValueError(f"rows({len(df)}) < seq_len({self.seq_len})")

        hist = df.iloc[-self.seq_len:].copy()

        # V3: Vx, Vy 계산
        cog_rad = np.radians(hist["cog"].values)
        sog = hist["sog"].values
        hist["vx"] = sog * np.sin(cog_rad)  # 동쪽 방향
        hist["vy"] = sog * np.cos(cog_rad)  # 북쪽 방향

        # Categorical ID
        if start_area is None:
            start_area = hist["start_area"].iloc[0] if "start_area" in hist.columns else "unknown"
        if end_area is None:
            end_area = hist["end_area"].iloc[0] if "end_area" in hist.columns else "unknown"

        self._current_start_id = self._get_start_area_id(start_area)
        self._current_end_id = self._get_end_area_id(end_area)

        return hist

    def _clamp_to_bounds(self, lat, lon, prev_lat, prev_lon):
        """항로 범위 벗어나면 보정"""
        if self.lat_bounds is None or self.lon_bounds is None:
            return lat, lon, False

        out_of_bounds = False
        clamped_lat = lat
        clamped_lon = lon

        if lat < self.lat_bounds[0]:
            clamped_lat = self.lat_bounds[0]
            out_of_bounds = True
        elif lat > self.lat_bounds[1]:
            clamped_lat = self.lat_bounds[1]
            out_of_bounds = True

        if lon < self.lon_bounds[0]:
            clamped_lon = self.lon_bounds[0]
            out_of_bounds = True
        elif lon > self.lon_bounds[1]:
            clamped_lon = self.lon_bounds[1]
            out_of_bounds = True

        if out_of_bounds:
            clamped_lat = (prev_lat + clamped_lat) / 2
            clamped_lon = (prev_lon + clamped_lon) / 2

        return clamped_lat, clamped_lon, out_of_bounds

    def predict_multi_from_df(self, df, n_steps=80,
                              start_area=None, end_area=None,
                              sog_clip=(0.0, 35.0),
                              sog_min_ratio=0.7,
                              use_model_latlon=False,
                              enforce_bounds=True,
                              enforce_sea_boundary=True):
        """
        다중 스텝 예측 (1분 간격) - V3

        Parameters:
        -----------
        df: 입력 DataFrame (datetime, lat, lon, sog, cog 필요)
        n_steps: 예측 스텝 수
        start_area, end_area: Categorical 값 (None이면 df에서 추출)
        sog_clip: SOG 클리핑 범위 (min, max)
        sog_min_ratio: 입력 데이터 평균 SOG 대비 최소 비율
        enforce_sea_boundary: 해역 경계 적용 여부
        """
        hist = self._prepare_hist(df, start_area, end_area)

        # V3 수치형 피처: lat, lon, vx, vy
        X_num = hist[["lat", "lon", "vx", "vy"]].values.astype(np.float32)
        Xn_num = (X_num - self.x_mean) / self.x_std

        # 입력 데이터의 평균 SOG
        input_sog_mean = float(hist["sog"].mean())
        sog_min_threshold = input_sog_mean * sog_min_ratio

        # V3 Categorical 피처: 3개 (start, end, grid)
        X_cat = np.zeros((self.seq_len, 3), dtype=np.int64)
        X_cat[:, 0] = self._current_start_id
        X_cat[:, 1] = self._current_end_id

        # 격자 ID는 각 위치마다 다름
        for i in range(self.seq_len):
            X_cat[i, 2] = self._get_grid_id(hist["lat"].iloc[i], hist["lon"].iloc[i])

        cur_lat = float(hist["lat"].iloc[-1])
        cur_lon = float(hist["lon"].iloc[-1])
        last_time = hist["datetime"].iloc[-1]

        # 이전 속도 벡터
        prev_vx = float(hist["vx"].iloc[-1])
        prev_vy = float(hist["vy"].iloc[-1])

        preds = []
        bound_violations = 0
        sea_boundary_corrections = 0

        for _ in range(int(n_steps)):
            x_num_t = torch.from_numpy(Xn_num).unsqueeze(0).to(self.device)
            x_cat_t = torch.from_numpy(X_cat).unsqueeze(0).to(self.device)

            with torch.inference_mode():
                y_hat_n = self.model(x_num_t, x_cat_t).squeeze(0).cpu().numpy().astype(np.float32)

            y_hat = y_hat_n * self.y_std.squeeze(0) + self.y_mean.squeeze(0)
            pred_lat_m, pred_lon_m, pred_vx, pred_vy = y_hat.tolist()

            # Vx, Vy로부터 SOG, COG 계산
            pred_sog = np.sqrt(pred_vx**2 + pred_vy**2)
            pred_cog = np.degrees(np.arctan2(pred_vx, pred_vy))
            pred_cog = (pred_cog + 360) % 360

            # SOG 클리핑
            effective_sog_min = max(sog_clip[0], sog_min_threshold)
            pred_sog = float(np.clip(pred_sog, effective_sog_min, sog_clip[1]))

            if use_model_latlon:
                next_lat, next_lon = float(pred_lat_m), float(pred_lon_m)
            else:
                next_lat, next_lon = step_latlon_dead_reckoning(cur_lat, cur_lon, pred_sog, pred_cog, dt_minutes=1)

            if enforce_bounds:
                next_lat, next_lon, violated = self._clamp_to_bounds(next_lat, next_lon, cur_lat, cur_lon)
                if violated:
                    bound_violations += 1

            # 해역 경계 적용
            if enforce_sea_boundary and self.sea_mask is not None:
                if not is_valid_sea_position(next_lat, next_lon, self.sea_mask):
                    corrected_lat, corrected_lon = find_nearest_valid_position(
                        next_lat, next_lon, cur_lat, cur_lon, self.sea_mask
                    )
                    next_lat, next_lon = corrected_lat, corrected_lon
                    sea_boundary_corrections += 1

            last_time = last_time + pd.Timedelta(minutes=1)
            preds.append([last_time, next_lat, next_lon, pred_sog, pred_cog])

            cur_lat, cur_lon = next_lat, next_lon

            # 시퀀스 업데이트 (V3: Vx, Vy)
            # SOG가 클리핑된 경우 Vx, Vy도 조정
            cog_rad = np.radians(pred_cog)
            new_vx = pred_sog * np.sin(cog_rad)
            new_vy = pred_sog * np.cos(cog_rad)

            new_row_num = np.array([cur_lat, cur_lon, new_vx, new_vy], dtype=np.float32)
            new_row_num_n = (new_row_num - self.x_mean.squeeze(0)) / self.x_std.squeeze(0)
            Xn_num = np.vstack([Xn_num[1:], new_row_num_n])

            # Categorical 업데이트 (격자 ID만 변경)
            new_grid_id = self._get_grid_id(cur_lat, cur_lon)
            new_cat_row = np.array([self._current_start_id, self._current_end_id, new_grid_id], dtype=np.int64)
            X_cat = np.vstack([X_cat[1:], new_cat_row])

        if bound_violations > 0:
            print(f"[InferenceV3] 경고: {bound_violations}회 항로 범위 이탈 보정됨")
        if sea_boundary_corrections > 0:
            print(f"[InferenceV3] 경고: {sea_boundary_corrections}회 육지 이탈 보정됨")

        return pd.DataFrame(preds, columns=["datetime", "pred_lat", "pred_lon", "pred_sog", "pred_cog"])


if __name__ == "__main__":
    print("TrajectoryInferenceV3 - 속도 벡터 (Vx, Vy) 지원")
    print("사용 예시:")
    print('  inf = TrajectoryInferenceV3("model/model_tft_v3.pth", "model/scaler_tft_v3.npz")')
    print('  preds = inf.predict_multi_from_df(df, n_steps=30, start_area="남쪽진입", end_area="여수정박지B")')
    print()
    print("V2 호환:")
    print('  inf = TrajectoryInferenceV2("model_v2.pth", "scaler_v2.npz")')
    print('  preds = inf.predict_multi_from_df(df, n_steps=30, mmsi="209110000")')
