import torch
import torch.nn as nn
from typing import Optional

class IMURecurrentNetwork(nn.Module):
    def __init__(self, input_dim: int = 17, hidden_dim: int = 128, feedback_dim: int = 32, output_dim: int = 3):
        """
        Auto-regressive Recursive Neural Network for IMU position estimation.
        
        Args:
            input_dim: Dimension of sensor input (17)
            hidden_dim: Hidden state size of the GRU
            feedback_dim: Size of the bottleneck vector fed back into the input
            output_dim: Predicted delta-position (3)
        """
        super(IMURecurrentNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.feedback_dim = feedback_dim
        
        # The GRU takes sensor data + previous feedback vector
        self.gru = nn.GRUCell(input_dim + feedback_dim, hidden_dim)
        
        # Bottleneck layer: maps GRU hidden state to the feedback vector
        self.bottleneck = nn.Linear(hidden_dim, feedback_dim)
        
        # Output layer: maps GRU hidden state to delta-position
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass implementing an auto-regressive recursive loop.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            h0: Initial hidden state (batch_size, hidden_dim)
            
        Returns:
            outputs: Predicted delta positions for each step (batch_size, seq_len, output_dim)
        """
        batch_size, seq_len, _ = x.size()
        device = x.device

        if h0 is None:
            h0 = torch.zeros(batch_size, self.hidden_dim).to(device)
        
        # Initial feedback vector (f_{t-1} for t=0)
        f_t = torch.zeros(batch_size, self.feedback_dim).to(device)
        h_t = h0
        
        outputs = []
        
        for t in range(seq_len):
            # Construct current input: concat(SensorData_t, Feedback_{t-1})
            z_t = torch.cat([x[:, t, :], f_t], dim=-1)
            
            # Update recurrent state
            h_t = self.gru(z_t, h_t)
            
            # Generate feedback for next timestep and output prediction
            f_t = self.bottleneck(h_t)
            out_t = self.output_layer(h_t)
            
            outputs.append(out_t)
        
        return torch.stack(outputs, dim=1)

    def predict_step(self, sensor_input: torch.Tensor, h_prev: torch.Tensor, f_prev: torch.Tensor):
        """
        Single step inference for real-time usage.
        """
        z_t = torch.cat([sensor_input, f_prev], dim=-1)
        h_t = self.gru(z_t, h_prev)
        f_t = self.bottleneck(h_t)
        out_t = self.output_layer(h_t)
        return out_t, h_t, f_t
