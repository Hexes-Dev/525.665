import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from datetime import timedelta
from src.data import data_tools

class IMUDatasetV2(Dataset):
    def __init__(self, db_path: str, calibration_file: str = "data/calibration.json", 
                 norm_params_path: str = "src/ml/norm_params.json", 
                 window_duration: float = 10.0, sensor_list: list = None, anchors: list = None):
        self.db = data_tools.Database(db_path, calibration_file=calibration_file)
        self.window_duration = window_duration
        self.sensor_list = sensor_list or data_tools.SENSOR_LIST
        
        # Load normalization parameters
        with open(norm_params_path, 'r') as f:
            self.norm_params = json.load(f)

        if anchors is not None:
            self.anchors = anchors
        else:
            # Identify Zero-Velocity Anchors from GPS
            print("Identifying zero-velocity anchors...")
            gps_readings = self.db.get_gps_readings()
            if not isinstance(gps_readings, list):
                gps_readings = [gps_readings]
            
            self.anchors = []
            timeframe_start = None
            for idx, reading in enumerate(gps_readings):
                if reading.speed <= 0.01 and timeframe_start is None:
                    timeframe_start = reading.timestamp
                elif reading.speed > 0.01 and timeframe_start is not None:
                    self.anchors.append(timeframe_start)
                    timeframe_start = None
            
            if timeframe_start is not None:
                self.anchors.append(timeframe_start)

    def __len__(self) -> int:
        return len(self.anchors)

    def _normalize(self, value, sensor_name, key):
        """Applies Z-score normalization using per-sensor or global params."""
        if sensor_name in self.norm_params["per_sensor"]:
            stats = self.norm_params["per_sensor"][sensor_name][key]
        else:
            stats = self.norm_params["global"][key]

        if isinstance(stats, dict): # For acc, gyr, mag (arrays)
            mean = np.array(stats["mean"])
            std = np.array(stats["std"])
            return (value - mean) / (std + 1e-6)
        else:
            if isinstance(stats, dict):
                return (value - stats["mean"]) / (stats["std"] + 1e-6)
            return value

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        t0 = self.anchors[idx]
        tn = t0 + timedelta(seconds=self.window_duration)

        # 1. Pull IMU data for the window
        imu_readings = self.db.get_imu_readings(start_datetime=t0, end_datetime=tn)
        if not imu_readings:
            return torch.zeros((1, 17)), torch.zeros((1, 3))
        if not isinstance(imu_readings, list):
            imu_readings = [imu_readings]
        
        # Sort IMU readings by timestamp to ensure chronological order across sensors
        imu_readings.sort(key=lambda x: x.timestamp)

        # 2. Pull GPS data for the window to calculate targets
        gps_readings = self.db.get_gps_readings(start_datetime=t0, end_datetime=tn)
        if not gps_readings:
            return torch.zeros((1, 17)), torch.zeros((1, 3))
        if not isinstance(gps_readings, list):
            gps_readings = [gps_readings]

        # Origin for NED conversion (first GPS point in the window)
        origin = gps_readings[0]
        origin_lat = data_tools.ddmm_to_decimal(origin.latitude)
        origin_lon = data_tools.ddmm_to_decimal(origin.longitude)
        origin_latlonalt = (origin_lat, origin_lon, origin.altitude)

        features = []
        targets = []
        
        prev_timestamp = None
        prev_ned_pos = np.array([0.0, 0.0, 0.0]) # Relative to origin

        for r in imu_readings:
            # --- Feature Engineering ---
            norm_acc = self._normalize(r.acc, r.sensor_name, "acc")
            norm_gyr = self._normalize(r.gyr, r.sensor_name, "gyr")
            norm_mag = self._normalize(r.mag, r.sensor_name, "mag")
            norm_tmp = self._normalize(r.tmp, r.sensor_name, "tmp")

            # One-hot encoding for sensors
            one_hot = np.zeros(len(self.sensor_list))
            if r.sensor_name in self.sensor_list:
                one_hot[self.sensor_list.index(r.sensor_name)] = 1.0

            # Time delta
            delta_t = 0.0
            if prev_timestamp is not None:
                delta_t = (r.timestamp - prev_timestamp).total_seconds()
            
            feat_vec = np.concatenate([
                norm_acc, norm_gyr, norm_mag, [norm_tmp], 
                one_hot, [delta_t]
            ])
            features.append(feat_vec)

            # --- Target Engineering (NED Delta) ---
            closest_gps = min(gps_readings, key=lambda g: abs((g.timestamp - r.timestamp).total_seconds()))
            curr_lat = data_tools.ddmm_to_decimal(closest_gps.latitude)
            curr_lon = data_tools.ddmm_to_decimal(closest_gps.longitude)
            curr_ned_pos = data_tools.latlon_to_ned(
                curr_lat, curr_lon, closest_gps.altitude,
                *origin_latlonalt
            )
            
            delta_pos = curr_ned_pos - prev_ned_pos
            targets.append(delta_pos)

            prev_timestamp = r.timestamp
            prev_ned_pos = curr_ned_pos

        return torch.tensor(np.array(features), dtype=torch.float32), \
               torch.tensor(np.array(targets), dtype=torch.float32)

def collate_fn(batch):
    xs = [item[0] for item in batch]
    ys = [item[1] for item in batch]
    xs_padded = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0.0)
    ys_padded = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=0.0)
    return xs_padded, ys_padded

def get_dataloader(db_path: str, batch_size: int = 32, window_duration: float = 10.0) -> DataLoader:
    dataset = IMUDatasetV2(db_path, window_duration=window_duration)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

def split_imu_datasets(db_path: str, train_split: float = 0.8, seed: int = None, **kwargs) -> tuple[IMUDatasetV2, IMUDatasetV2]:
    temp_ds = IMUDatasetV2(db_path, **kwargs)
    all_anchors = temp_ds.anchors
    if seed:
        np.random.seed(seed)
    np.random.shuffle(all_anchors)
    split_idx = int(len(all_anchors) * train_split)
    train_anchors = all_anchors[:split_idx]
    test_anchors = all_anchors[split_idx:]
    train_ds = IMUDatasetV2(db_path, anchors=train_anchors, **kwargs)
    test_ds = IMUDatasetV2(db_path, anchors=test_anchors, **kwargs)
    return train_ds, test_ds
