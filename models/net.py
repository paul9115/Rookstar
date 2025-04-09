import torch
import torch.nn as nn


class RookstarNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 4096),  # later we'll mask to legal moves
        )

        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value
