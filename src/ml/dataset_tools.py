from enum import Enum
import json
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.data.data_tools import Database, GPSReading, SENSOR_LIST, ddmm_to_decimal
from datetime import datetime, timedelta
from src.ml.models import ModelType


class IMUDataset(Dataset):
    def __init__(self, db_path: str = "data/dataset.db", calibration_path: str = "data/calibration.json",
                 normalize_params: bool = True, normalization_params_path: str = "src/ml/norm_params.json", window_duration: float = 2.0, 
                 sensor_list: list[str] = None, model_type: ModelType = ModelType.KINEMATICS, windows: list[tuple[datetime, datetime]] = None):
        
        self.db_path = db_path
        self.calibration_path = calibration_path
        self.normalize_params = normalize_params
        self.normalize_params_path = normalization_params_path

        self.model_type = model_type
        self.db: Database = Database(db_path, calibration_path)
        self.window_duration = window_duration
        self.sensor_list = sensor_list or SENSOR_LIST or []

        # Load normalization parameters
        with open(self.normalization_params_path, 'r') as f:
            self.normalization_params = json.load(f)

        # Retrieve windows
        if windows:
            self.windows = windows
        else:
            print("Identifying zero-velocity anchors.")

            zero_velocity_readings: List[GPSReading] = self.db.get_gps_readings(max_speed=0.01)

            self.windows: List[Tuple[datetime, datetime]] = []

            for reading in zero_velocity_readings:
                window = self.db.get_gps_readings(start_datetime=reading.timestamp, end_datetime=reading.timestamp + timedelta(seconds=(self.window_duration + 0.5)))
                if (window[-1].timestamp - window[0].timestamp).total_seconds() >= self.window_duration:
                    new_window = (window[0].timestamp, window[-1].timestamp)
                    self.windows.append(new_window)

            print(f"Found {len(self.windows)} with sufficient length.")


    def subsample_data(self):

        for window in list(self.windows):
            gps_readings = self.db.get_gps_readings(start_datetime=window[0].timestamp, end_datetime=window[-1].timestamp)
            if len(gps_readings) > 2:
                for idx in range(len(gps_readings) - 2):
                    sub_window = (gps_readings[0].timestamp, gps_readings[idx].timestamp)
                    self.windows.append(sub_window)

    def __len__(self) -> int:
        return len(self.windows)
    
    def _normalize(self, value, sensor_name, key):
        """Applies Z-score normalization using per-sensor or global params."""
        if sensor_name in self.normalization_params["per_sensor"]:
            stats = self.normalization_params["per_sensor"][sensor_name][key]
        else:
            stats = self.normalization_params["global"][key]

        if isinstance(stats, dict): # For acc, gyr, mag (arrays)
            mean = np.array(stats["mean"])
            std = np.array(stats["std"])
            return (value - mean) / (std + 1e-6)
        else:
            if isinstance(stats, dict):
                return (value - stats["mean"]) / (stats["std"] + 1e-6)
            return value
        
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        t0, tn = self.windows[idx]

        imu_readings = self.db.get_imu_readings(start_datetime=t0, end_datetime=tn)
        imu_readings.sort(key=lambda x: x.timestamp)

        gps_readings = self.db.get_gps_readings(start_datetime=t0, end_datetime=tn)
        gps_readings.sort(key=lambda x: x.timestamp)

        origin = gps_readings[0]
        origin_lat = ddmm_to_decimal(origin.latitude)
        origin_lon = ddmm_to_decimal(origin.longitude)
        origin_lat_lon_alt = (origin_lat, origin_lon, origin.altitude)

        # Normalize IMU data
        if self.normalize_params:

            for r in imu_readings:
                r.acc = self._normalize(r.acc, r.sensor_name, "acc")
                r.gyr = self._normalize(r.gyr, r.sensor_name, "gyr")
                r.mag = self._normalize(r.mag, r.sensor_name, "mag")
                r.tmp = self._normalize(r.tmp, r.sensor_name, "tmp")

        # Return the correct output depending on the type of model selected
        if self.model_type is ModelType.KINEMATICS:
            pass
        else:
            return torch.tensor(np.zeros(0), dtype=torch.float32), torch.tensor(np.zeros(0), dtype=torch.float32)

    def test_train_split(self, train_split: float = 0.5, seed: int = 0):
        np.random.seed(seed)
        np.random.shuffle(self.windows)

        split_idx = int(len(self.windows) * train_split)
        train_windows = self.windows[:split_idx]
        test_windows = self.windows[split_idx:]
        train_dataset = IMUDataset(
            self.db_path, 
            self.calibration_path, 
            self.normalize_params, 
            self.normalize_params_path, 
            self.window_duration, 
            self.sensor_list,
            self.model_type,
            train_windows,
        )
        test_dataset = IMUDataset(
            self.db_path,
            self.calibration_path, 
            self.normalize_params, 
            self.normalize_params_path, 
            self.window_duration, 
            self.sensor_list,
            self.model_type,
            test_windows,
        )
        return train_dataset, test_dataset