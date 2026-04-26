import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

def quaternion_to_rotation_matrix(q):
    """
    Converts a batch of quaternions [w, x, y, z] to rotation matrices.
    q: (batch_size, 4)
    returns: (batch_size, 3, 3)
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    # Precompute squares
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

class SensorEncoder(nn.Module):
    def __init__(self, input_dim=17, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)

class IMUKinematicNetwork(nn.Module):
    def __init__(self, input_dim=17, hidden_dim=128, latent_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # 1. Shared Encoder to handle sensor biases/noise
        self.encoder = SensorEncoder(input_dim, latent_dim)
        
        # 2. Recurrent State Core (tracks Orientation and Velocity latent states)
        # Input: Latent features + dt
        self.gru = nn.GRUCell(latent_dim + 1, hidden_dim)
        
        # 3. Physical State Heads
        self.q_head = nn.Linear(hidden_dim, 4)   # Predicted Quaternion [w, x, y, z]
        self.v_head = nn.Linear(hidden_dim, 3)   # Predicted World Velocity [vx, vy, vz]
        
    def forward(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with Kinematic Integration logic.
        x: (batch_size, seq_len, 17)
        """
        batch_size, seq_len, _ = x.size()
        device = x.device
        
        if h0 is None:
            h0 = torch.zeros(batch_size, self.hidden_dim).to(device)
        
        h_t = h0
        outputs = []
        
        for t in range(seq_len):
            # Current input slice
            x_t = x[:, t, :] # (batch_size, 17)
            
            # Extract components for physics integration
            # Input layout: [acc(3), gyr(3), mag(3), temp(1), onehot(6), dt(1)]
            acc_norm = x_t[:, 0:3] 
            dt = x_t[:, -1:] # (batch_size, 1)
            
            # Encode sensor data into latent space
            latent_z = self.encoder(x_t) # (batch_size, latent_dim)
            
            # Update State Core
            gru_input = torch.cat([latent_z, dt], dim=-1)
            h_t = self.gru(gru_input, h_t)
            
            # Decode Physical States
            q_raw = self.q_head(h_t)
            # Normalize to unit quaternion
            q = F.normalize(q_raw, p=2, dim=-1)
            v_world = self.v_head(h_t)
            
            # Physics-Informed Integration:
            # 1. Compute Rotation Matrix from predicted orientation
            R = quaternion_to_rotation_matrix(q) # (batch_size, 3, 3)
            
            # 2. Project Normalized Accel to World Frame (Guided feature transformation)
            # a_world = R * a_norm
            acc_world = torch.bmm(R, acc_norm.unsqueeze(-1)).squeeze(-1) # (batch_size, 3)
            
            # 3. Kinematic Integration: DeltaPos = v*dt + 0.5*a*dt^2
            # We treat the "normalized" accel as a guided signal for acceleration
            dp = v_world * dt + 0.5 * acc_world * (dt**2)
            
            outputs.append(dp)
            
        return torch.stack(outputs, dim=1)

    def predict_step(self, sensor_input: torch.Tensor, h_prev: torch.Tensor):
        """
        Single step inference for real-time usage.
        sensor_input: (17,)
        h_prev: (hidden_dim,)
        """
        # Ensure batch dim
        x = sensor_input.unsqueeze(0) if sensor_input.dim() == 1 else sensor_input
        h = h_prev.unsqueeze(0) if h_prev.dim() == 1 else h_prev
        
        acc_norm = x[:, 0:3]
        dt = x[:, -1:]
        
        latent_z = self.encoder(x)
        h_t = self.gru(torch.cat([latent_z, dt], dim=-1), h)
        
        q = F.normalize(self.q_head(h_t), p=2, dim=-1)
        v_world = self.v_head(h_t)
        
        R = quaternion_to_rotation_matrix(q)
        acc_world = torch.bmm(R, acc_norm.unsqueeze(-1)).squeeze(-1)
        dp = v_world * dt + 0.5 * acc_world * (dt**2)
        
        return dp, h_t
