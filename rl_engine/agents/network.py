"""Q-network architecture."""

import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int = 5, hidden_sizes: tuple[int, ...] = (128, 128)):
        super().__init__()
        if state_size <= 0 or action_size <= 0:
            raise ValueError("state_size and action_size must be positive")
        layers: list[nn.Module] = []
        input_size = state_size
        for hidden_size in hidden_sizes:
            layers.extend((nn.Linear(input_size, hidden_size), nn.ReLU()))
            input_size = hidden_size
        layers.append(nn.Linear(input_size, action_size))
        self.model = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.model(state)
