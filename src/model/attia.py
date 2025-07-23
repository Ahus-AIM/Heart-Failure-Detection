import torch
import torch.nn as nn
from torch import Tensor


class AttiaNetwork(nn.Module):
    def __init__(self, num_channels: int) -> None:
        super(AttiaNetwork, self).__init__()
        self.num_channels: int = num_channels

        self.temporal_net: nn.Sequential = nn.Sequential(
            nn.Conv1d(1, 16, 5),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 16, 5),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 32, 3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )

        self.spatial_net: nn.Sequential = nn.Sequential(
            nn.Conv2d(self.num_channels, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.fc: nn.Sequential = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(16 * 64 * 18, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: shape [B, C, L] where C = self.num_channels
        x_temporal: Tensor = torch.empty(0, device=x.device)
        for i in range(self.num_channels):
            x_channel: Tensor = x[:, i, :].unsqueeze(1)  # shape [B, 1, L]
            x_channel = self.temporal_net(x_channel)  # shape [B, 64, T]
            x_channel = x_channel.unsqueeze(1)  # shape [B, 1, 64, T]
            if i == 0:
                x_temporal = x_channel
            else:
                x_temporal = torch.cat((x_temporal, x_channel), dim=1)  # shape [B, C, 64, T]

        x_spatial: Tensor = self.spatial_net(x_temporal)  # shape [B, 16, 64, T]
        x_out: Tensor = self.fc(x_spatial)  # shape [B, 1]
        return x_out
