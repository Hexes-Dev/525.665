# IMU State Estimation ML Project

This project implements and compares machine learning models for estimating state from IMU sensor data, complementing traditional methods like Extended Kalman Filters (EKF).

## Project Structure

- `src/ml/`: Model implementations and training scripts.
  - `imu_model.py`: Baseline GRU Recurrent Network (V1).
  - `imu_model_v2.py`: Kinematic Network (V2).
- `src/data/`: Sensor log reading, calibration, and preprocessing tools.
- `src/ekf/`: Utilities for Extended Kalman Filter implementation.
- `configs/`: JSON configuration files for hyperparameters and data paths.
- `data/`: Dataset databases and calibration files.

## Getting Started

### Prerequisites
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Training Models
Training is now unified in a single script. Model and dataset selection are handled via the configuration file.

```bash
python src/ml/train.py --config configs/baseline.json
```

**Configuration keys for model selection:**
- `model_type`: `"baseline"` (V1 GRU) or `"kinematic"` (V2 Kinematic Network).
- `dataset_type`: `"v1"` or `"v2"`.

Optional flags:
- `--mode`: `gui`, `tui` (default), or `plain` for logging format.
- `--output-dir`: Base directory for results (defaults to `outputs/`). Results are saved in subdirectories by model type.

### Verification
Run a smoke test to ensure both models can perform a forward and backward pass with the current dataset:
```bash
python src/ml/smoke_test.py
```

## Configuration
Hyperparameters, data paths, and normalization parameters are managed in `configs/*.json`. Key fields include:
- `db_path`: Path to the sensor dataset database.
- `calibration_file`: Path to calibration constants.
- `norm_params_path`: Path to normalization parameters.
