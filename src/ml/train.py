import os
import sys
import json
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.ml.imu_data_loader import IMUDataset, get_dataloader, split_imu_datasets
from src.ml.imu_model import IMURecurrentNetwork

class LivePlotter:
    def __init__(self, config_name):
        self.config_name = config_name
        plt.ion() # Interactive mode on
        self.fig, self.axs = plt.subplots(2, 2, figsize=(12, 8))
        self.axs = self.axs.flatten()
        
        self.metrics = {
            "loss": [],
            "time_per_batch": [],
            "inference_time": []
        }
        
        # Initialize plots
        self.axs[0].set_title("Training Loss")
        self.axs[1].set_title("Time per Batch (s)")
        self.axs[2].set_title("Inference Time (ms)")
        self.axs[3].set_title("Metric Summary")

    def update(self, loss, batch_time, inf_time):
        self.metrics["loss"].append(loss)
        self.metrics["time_per_batch"].append(batch_time)
        self.metrics["inference_time"].append(inf_time * 1000)

        for i, (name, data) in enumerate(self.metrics.items()):
            self.axs[i].clear()
            self.axs[i].plot(data, color='blue')
            self.axs[i].set_title(f"{name}")
            self.axs[i].set_xlabel("Batch")
            self.axs[i].set_ylabel("Value")

        plt.tight_layout()
        plt.pause(0.01)

    def save_final(self):
        plt.ioff()
        filename = f"outputs/{self.config_name}_training_plot.png"
        os.makedirs("outputs", exist_ok=True)
        plt.savefig(filename)
        print(f"Final plots saved to {filename}")

def get_component(module, name, params):
    if name is None: return None
    cls = getattr(module, name)
    return cls(**params)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON file")
    args = parser.parse_args()

    # Load Config
    with open(args.config, 'r') as f:
        cfg = json.load(f)
    
    base_name = os.path.splitext(os.path.basename(args.config))[0]
    
    # Setup Data
    train_ds, test_ds = split_imu_datasets(
        db_path=cfg["db_path"], 
        train_split=cfg.get("train_split", 0.8),
        calibration_file=cfg["calibration_file"],
        norm_params_path=cfg["norm_params_path"],
        window_duration=cfg["window_duration"]
    )
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False)

    # Setup Model
    model = IMURecurrentNetwork(
        input_dim=17, 
        hidden_dim=cfg["hidden_dim"], 
        feedback_dim=cfg["feedback_dim"]
    )
    
    # Loss and Optimizer
    loss_fn_cls = getattr(nn.Module, cfg["loss_fn"]) if hasattr(nn.Module, cfg["loss_fn"]) else getattr(nn, cfg["loss_fn"])
    criterion = loss_fn_cls()
    
    optimizer_cls = getattr(optim, cfg["optimizer"])
    optimizer = optimizer_cls(model.parameters(), lr=cfg["lr"], **cfg.get("optimizer_params", {}))
    
    scheduler_cls = getattr(optim.lr_scheduler, cfg["scheduler"]) if cfg["scheduler"] else None
    scheduler = scheduler_cls(optimizer, **cfg.get("scheduler_params", {})) if scheduler_cls else None

    # Visualization
    try:
        plotter = LivePlotter(base_name)
    except Exception as e:
        print(f"Warning: Could not initialize live plotter ({e}). Progress will be logged to console.")
        plotter = None

    # Training Loop
    model.train()
    history = {"loss": [], "test_loss": []}
    start_time = time.time()

    for epoch in range(cfg["epochs"]):
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}")
        
        for batch_idx, (x, y) in enumerate(pbar):
            optimizer.zero_grad()
            
            # Measure inference/forward time
            t0 = time.time()
            outputs = model(x)
            t_inf = time.time() - t0
            
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            batch_time = time.time() - t0 # Total batch time approx
            epoch_loss += loss.item()
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "inf_t": f"{t_inf*1000:.2f}ms"})
            
            if plotter:
                plotter.update(loss.item(), batch_time, t_inf)

        avg_train_loss = epoch_loss / len(train_loader)
        history["loss"].append(avg_train_loss)
        
        # Validation Step
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for x_test, y_test in test_loader:
                outputs_test = model(x_test)
                test_loss += criterion(outputs_test, y_test).item()
        avg_test_loss = test_loss / len(test_loader)
        history["test_loss"].append(avg_test_loss)
        model.train()

        if scheduler:
            scheduler.step()
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}")

    total_time = time.time() - start_time
    
    # Save Results
    results = {
        "final_train_loss": history["loss"][-1],
        "initial_train_loss": history["loss"][0],
        "final_test_loss": history["test_loss"][-1],
        "total_training_time": total_time,
        "epochs": cfg["epochs"],
        "config": cfg
    }
    
    os.makedirs("outputs", exist_ok=True)
    with open(f"outputs/{base_name}_results.json", "w") as f:
        json.dump(results, f, indent=4)

    if plotter:
        plotter.save_final()

    print(f"Training complete. Results saved to outputs/{base_name}_results.json")

if __name__ == "__main__":
    train()
