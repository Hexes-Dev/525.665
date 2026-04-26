from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class ModelType(Enum):
    KINEMATICS = 1

def quaternion_to_rotation_matrix(q):
    """Converts a batch of quaternions [w, x, y, z] to rotation matrices."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    batch_size = q.shape[0]
    R = torch.zeros((batch_size, 3, 3), device=q.device)
    R[:, 0, 0] = 1 - 2 * (yy + zz)
    R[:, 0, 1] = 2 * (xy - wz)
    R[:, 0, 2] = 2 * (xz + wy)
    R[:, 1, 0] = 2 * (xy + wz)
    R[:, 1, 1] = 1 - 2 * (xx + zz)
    R[:, 1, 2] = 2 * (yz - wx)
    R[:, 2, 0] = 2 * (xz - wy)
    R[:, 2, 1] = 2 * (yz + wx)
    R[:, 2, 2] = 1 - 2 * (xx + yy)
    return R

class IMUKinematicNetwork(nn.Module):
    def __init__(self, input_dim=17, hidden_dim=128, latent_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU()
        )
        
        self.gru = nn.GRUCell(latent_dim + 1, hidden_dim)
        self.q_head = nn.Linear(hidden_dim, 4)
        self.v_head = nn.Linear(hidden_dim, 3)
        self.correction_head = nn.Linear(hidden_dim, 3)

    def forward(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        device = x.device
        h_t = h0 if h0 is not None else torch.zeros(batch_size, self.hidden_dim).to(device)
        total_dp = torch.zeros(batch_size, 3).to(device)

        for t in range(seq_len):
            x_t = x[:, t, :]
            dt = x_t[:, -1:]
            latent_z = self.encoder(x_t)
            gru_input = torch.cat([latent_z, dt], dim=-1)
            h_t = self.gru(gru_input, h_t)
            
            q = F.normalize(self.q_head(h_t), p=2, dim=-1)
            v_world = self.v_head(h_t)
            acc_norm = x_t[:, 0:3]
            R = quaternion_to_rotation_matrix(q)
            acc_world = torch.bmm(R, acc_norm.unsqueeze(-1)).squeeze(-1)
            
            dp_baseline = v_world * dt + 0.5 * acc_world * (dt**2)
            epsilon = self.correction_head(h_t)
            total_dp += (dp_baseline + epsilon)

        return total_dp
