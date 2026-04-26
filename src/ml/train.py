import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import yaml
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.ml.imu_data_loader import IMUDataset, get_dataloader, split_imu_datasets as split_v1, collate_fn as collate_v1
from src.ml.imu_model import IMURecurrentNetwork
from src.ml.imu_data_loader_v2 import IMUDatasetV2, split_imu_datasets as split_v2, collate_fn as collate_v2
from src.ml.imu_model_v2 import IMUKinematicNetwork
from src.ml.logger import TrainingLogger

# Registry for easy model and dataset selection
MODEL_CONFIG = {
    "baseline": {
        "model_cls": IMURecurrentNetwork,
        "dataset": {"split": split_v1, "collate": collate_v1},
    },
    "kinematic": {
        "model_cls": IMUKinematicNetwork,
        "dataset": {"split": split_v2, "collate": collate_v2},
    },
}

def get_component(module, name, params):
    if name is None: return None
    cls = getattr(module, name)
    return cls(**params)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--mode", type=str, default="gui", choices=["gui", "plain"], help="Output mode: gui or plain")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Base directory for training outputs")
    args = parser.parse_args()

    # Load Config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    model_type = cfg.get("model_type", "baseline")
    base_name = os.path.splitext(os.path.basename(args.config))[0]
    
    # Resolve Model and Dataset from Registry
    if model_type not in MODEL_CONFIG:
        raise ValueError(f"Unsupported model_type: {model_type}. Available: {list(MODEL_CONFIG.keys())}")
    
    config_entry = MODEL_CONFIG[model_type]
    model_cls = config_entry["model_cls"]
    ds_utils = config_entry["dataset"]

    # Setup Data
    train_ds, test_ds = ds_utils["split"](
        db_path=cfg["db_path"], 
        train_split=cfg.get("train_split", 0.8),
        seed=cfg.get("seed"),
        calibration_file=cfg["calibration_file"],
        norm_params_path=cfg["norm_params_path"],
        window_duration=cfg["window_duration"]
    )
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, collate_fn=ds_utils["collate"])
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, collate_fn=ds_utils["collate"])

    # Setup Model
    if model_type == "baseline":
        model = model_cls(
            input_dim=17, 
            hidden_dim=cfg["hidden_dim"], 
            feedback_dim=cfg["feedback_dim"]
        )
    elif model_type == "kinematic":
        model = model_cls(
            input_dim=17,
            hidden_dim=cfg.get("hidden_dim", 128),
            latent_dim=cfg.get("latent_dim", 64)
        )
    else:
        # Generic instantiation if params are provided in config
        model = model_cls(**cfg.get("model_params", {}))

    # Loss and Optimizer
    loss_fn_cls = getattr(nn.Module, cfg["loss_fn"]) if hasattr(nn.Module, cfg["loss_fn"]) else getattr(nn, cfg["loss_fn"])
    criterion = loss_fn_cls()
    
    optimizer_cls = getattr(optim, cfg["optimizer"])
    optimizer = optimizer_cls(model.parameters(), lr=cfg["lr"], **cfg.get("optimizer_params", {}))
    
    scheduler_cls = getattr(optim.lr_scheduler, cfg["scheduler"]) if cfg["scheduler"] else None
    scheduler = scheduler_cls(optimizer, **cfg.get("scheduler_params", {})) if scheduler_cls else None

    # Logger Setup
    logger = TrainingLogger(mode=args.mode, config_name=base_name)

    # Training Loop
    model.train()
    history = {"loss": [], "test_loss": []}
    start_time = time.time()

    for epoch in range(cfg["epochs"]):
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}")
        logger.set_progress_bar(pbar)
        
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
            
            logger.log("batch", loss=loss.item(), batch_time=batch_time, inf_time=t_inf, lr=optimizer.param_groups[0]['lr'])

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
        logger.log("epoch", epoch=epoch + 1, train_loss=avg_train_loss, test_loss=avg_test_loss)

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
    
    run_dir = os.path.join(args.output_dir, model_type)
    os.makedirs(run_dir, exist_ok=True)
    result_path = os.path.join(run_dir, f"{base_name}_results.yaml")
    with open(result_path, "w") as f:
        yaml.dump(results, f)

    logger.save_final()
    
    print(f"Training complete. Results saved to {result_path}")


if __name__ == "__main__":
    train()
