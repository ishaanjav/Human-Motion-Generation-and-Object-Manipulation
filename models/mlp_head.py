import torch
import torch.nn as nn

class MLPHead(nn.Module):
    def __init__(self, input_dim=524, hidden_dims=[1024, 512], output_dim=None):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Add hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
            
        # Add output layer
        if output_dim is not None:
            layers.append(nn.Linear(prev_dim, output_dim))
            
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.mlp(x)