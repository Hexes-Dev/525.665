import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.ml.imu_data_loader import IMUDataset, split_imu_datasets, collate_fn
from src.ml.imu_model import IMURecurrentNetwork
from src.ml.imu_data_loader_v2 import IMUDatasetV2, split_imu_datasets as split_v2, collate_fn as collate_v2
from src.ml.imu_model_v2 import IMUKinematicNetwork

def test_model(name, model, loader, criterion, optimizer):
    print(f"Testing {name}...")
    batch = next(iter(loader))
    x, y = batch
    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    print(f"{name} - Success! Loss: {loss.item():.4f}")

if __name__ == "__main__":
    # Config
    db_path = "data/dataset.db"
    cal_file = "data/calibration.json"
    norm_path = "src/ml/norm_params.json"
    batch_size = 4 # Small batch for smoke test
    
    criterion = nn.MSELoss()

    # Test Model V1
    train_ds, _ = split_imu_datasets(db_path=db_path, calibration_file=cal_file, norm_params_path=norm_path)
    loader_v1 = DataLoader(train_ds, batch_size=batch_size, collate_fn=collate_fn)
    model_v1 = IMURecurrentNetwork()
    opt_v1 = optim.Adam(model_v1.parameters(), lr=0.001)
    test_model("Baseline GRU", model_v1, loader_v1, criterion, opt_v1)

    # Test Model V2
    train_ds_v2, _ = split_v2(db_path=db_path, calibration_file=cal_file, norm_params_path=norm_path)
    loader_v2 = DataLoader(train_ds_v2, batch_size=batch_size, collate_fn=collate_v2)
    model_v2 = IMUKinematicNetwork()
    opt_v2 = optim.Adam(model_v2.parameters(), lr=0.001)
    test_model("Kinematic Network", model_v2, loader_v2, criterion, opt_v2)
