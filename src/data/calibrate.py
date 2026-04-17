from dataclasses import astuple
from typing import List

import numpy as np
from numpy import random

from src.data.data_utils import Database, IMUReading, estimate_level_correction, SENSOR_LIST, calibrate_magnetometer, \
    XYZ, xyz_array, array_xyz


def main():
    db = Database("../../data/data.db")

    # Load all gps records
    gps_readings = db.read_gps()
    print(f"Found {len(gps_readings)} GPS readings")

    zero_speed_timeframes = []
    for idx, reading in enumerate(gps_readings):
        if idx == len(gps_readings) - 1:
            continue

        next_reading = gps_readings[idx + 1]
        if reading.speed <= 0.01 and next_reading.speed <= 0.01:
            zero_speed_timeframes.append((reading.timestamp, next_reading.timestamp))

    print(f"Found {len(zero_speed_timeframes)} zero speed timeframes")

    zero_speed_imu: List[IMUReading] = []
    for timeframe in zero_speed_timeframes:
        start_time, end_time = timeframe
        new_imu_entries = db.read_imu(start_time=start_time, end_time=end_time)
        if not new_imu_entries:
            continue
        zero_speed_imu = zero_speed_imu + new_imu_entries

        print(f"\rFound {len(new_imu_entries)} new IMU entries between {start_time} and {end_time} -- Total: {len(zero_speed_imu)}", end="", flush=True)

    print(f"Total zero speed entries: {len(zero_speed_imu)}")

    sensor_rotation_matrix = {}
    for sensor in SENSOR_LIST:
        print(f"Solving orientation calibration for {sensor}")
        per_sensor_imu = [reading for reading in zero_speed_imu if reading.sensor_name == sensor]
        random_samples = random.choice(per_sensor_imu, size=int(0.75*len(per_sensor_imu)))
        sensor_rotation_matrix[sensor] = estimate_level_correction(random_samples, len(random_samples))
        print(sensor_rotation_matrix[sensor])

    sensor_magnetic_correction = {}
    for sensor in SENSOR_LIST:
        print(f"Reading entries for magnetic calibration of {sensor}")
        imu_readings = db.read_imu(sensor_name=sensor)

        print(f"Solving magnetic calibration for {sensor}")
        # Apply rotation matrix prior to magnetic calibration
        r_level = sensor_rotation_matrix[sensor]
        for idx, reading in enumerate(imu_readings):
            gyro = r_level @ np.array(astuple(reading.raw_gyro), dtype=float)
            accel = r_level @ np.array(astuple(reading.raw_accelerometer), dtype=float)
            mag = r_level @ np.array(astuple(reading.raw_magnetometer), dtype=float)

            imu_readings[idx].accelerometer = XYZ(*accel)
            imu_readings[idx].gyro = XYZ(*gyro)
            imu_readings[idx].magnetometer = XYZ(*mag)

        oriented_magnetometer_readings = [xyz_array(imu.magnetometer) for imu in imu_readings]
        sensor_magnetic_correction[sensor] = calibrate_magnetometer(oriented_magnetometer_readings)
        print(sensor_magnetic_correction[sensor])

        print(f"Applying calibration for {sensor}")
        for idx, reading in enumerate(imu_readings):
            offset = np.array(sensor_magnetic_correction[sensor]['offset'])
            scale_matrix = np.array(sensor_magnetic_correction[sensor]['scale_matrix'])

            calibrated_magnetometer = (raw_vector := xyz_array(reading.magnetometer) - offset) @ scale_matrix.T
            imu_readings[idx].magnetometer = array_xyz(calibrated_magnetometer)

            sensor_id = reading.sensor_name.split('_')[-1]
            sensor_type = "_".join(reading.sensor_type.split('_')[:-1])
            imu_readings[idx].sensor_id = sensor_id
            imu_readings[idx].sensor_type = sensor_type

        print(f"Updating database for {sensor}")
        batch_size = 10000
        for i in range(0, len(imu_readings), batch_size):
            batch = imu_readings[i: i + batch_size]
            db.write_imu(batch)

if __name__ == '__main__':
    main()