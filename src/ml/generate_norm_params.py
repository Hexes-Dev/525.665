import json
import numpy as np
import random
from src.data import data_tools

def calculate_stats(readings):
    """Helper to compute mean and std from a list of readings."""
    if not readings:
        return {"mean": [0, 0, 0], "std": [1, 1, 1]}
    
    accs = np.array([r.acc for r in readings])
    gyrs = np.array([r.gyr for r in readings])
    mags = np.array([r.mag for r in readings])
    tmps = np.array([r.tmp for r in readings])

    return {
        "acc": {"mean": np.mean(accs, axis=0).tolist(), "std": np.std(accs, axis=0).tolist()},
        "gyr": {"mean": np.mean(gyrs, axis=0).tolist(), "std": np.std(gyrs, axis=0).tolist()},
        "mag": {"mean": np.mean(mags, axis=0).tolist(), "std": np.std(mags, axis=0).tolist()},
        "tmp": {"mean": float(np.mean(tmps)), "std": float(np.std(tmps))}
    }

def main():
    db = data_tools.Database("data/dataset.db", calibration_file="data/calibration.json")
    
    print("Fetching GPS readings to identify sample points...")
    gps_readings = db.get_gps_readings()
    if not isinstance(gps_readings, list):
        gps_readings = [gps_readings]
    
    if not gps_readings:
        print("No GPS readings found.")
        return

    # Sample a limited number of timestamps to avoid DB overload
    num_samples = min(len(gps_readings), 200)
    sample_points = random.sample(gps_readings, num_samples)
    
    sensor_data = {name: [] for name in data_tools.SENSOR_LIST}
    all_imu_pool = []

    print(f"Sampling IMU data around {num_samples} points...")
    for i, gps in enumerate(sample_points):
        # Query a 1-second window around the GPS timestamp
        start_time = gps.timestamp
        # Note: In a real scenario we might want to be more precise with the end_datetime
        # but for sampling purposes, we'll just pull some records starting from this point.
        # We'll use a small range or rely on the DB limit if it existed, 
        # but here we just query based on start_datetime and hope it's not too huge.
        # Since get_imu_readings takes end_datetime, let's add a small offset.
        from datetime import timedelta
        end_time = start_time + timedelta(seconds=1)
        
        readings = db.get_imu_readings(start_datetime=start_time, end_datetime=end_time)
        if not readings:
            continue
        if not isinstance(readings, list):
            readings = [readings]

        for r in readings:
            if r.sensor_name in sensor_data:
                sensor_data[r.sensor_name].append(r)
            all_imu_pool.append(r)
        
        if (i + 1) % 20 == 0:
            print(f"Processed {i+1}/{num_samples} sample points...", end="\r")

    # Calculate Per-Sensor Normalization
    per_sensor_params = {}
    for sensor, data in sensor_data.items():
        per_sensor_params[sensor] = calculate_stats(data)

    # Calculate Global Normalization
    global_params = calculate_stats(all_imu_pool)

    final_params = {
        "per_sensor": per_sensor_params,
        "global": global_params
    }

    with open("src/ml/norm_params.json", "w") as f:
        json.dump(final_params, f, indent=4)
    
    print(f"\nNormalization parameters saved to src/ml/norm_params.json")
    print(f"Total IMU samples collected for stats: {len(all_imu_pool)}")

if __name__ == "__main__":
    main()
