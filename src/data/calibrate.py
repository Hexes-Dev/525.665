from dataclasses import astuple
from typing import List

import numpy as np
from numpy import random

from data_tools import Database, IMUReading, estimate_level_correction, SENSOR_LIST, calibrate_magnetometer


def main():
    db = Database("data/dataset.db")

    # Load all gps records
    gps_readings = db.get_gps_readings()
    gps_readings = sorted(gps_readings, key=lambda r: r.timestamp)
    print(f"Found {len(gps_readings)} GPS readings")

    timeframe_start = None
    zero_speed_seconds = 0
    zero_speed_timeframes = []
    for idx, reading in enumerate(gps_readings):
        if idx == len(gps_readings) - 1:
            continue

        if reading.speed <= 0.01 and timeframe_start is None:
            timeframe_start = idx

        if reading.speed > 0.01 and timeframe_start is not None:
            zero_speed_seconds += (gps_readings[idx-1].timestamp - gps_readings[timeframe_start].timestamp).total_seconds()
            zero_speed_timeframes.append((gps_readings[timeframe_start].timestamp, gps_readings[idx-1].timestamp))
            timeframe_start = None

    print(f"Found {len(zero_speed_timeframes)} zero speed timeframes -- {zero_speed_seconds} seconds of data.")

    zero_speed_imu: List[IMUReading] = []
    for idx, timeframe in enumerate(zero_speed_timeframes):
        start_time, end_time = timeframe
        new_imu_entries = db.get_imu_readings(start_datetime=start_time, end_datetime=end_time)
        if not new_imu_entries:
            continue
        zero_speed_imu = zero_speed_imu + new_imu_entries

        print(f"\033[KFound {len(new_imu_entries)} new IMU entries between {start_time} and {end_time} -- Total: {len(zero_speed_imu)}", end="\r", flush=True)

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
        imu_readings = db.get_imu_readings(sensor_name=sensor)

        print(f"Solving magnetic calibration for {sensor} using {len(imu_readings)} readings")
        # Apply rotation matrix prior to magnetic calibration
        r_level = sensor_rotation_matrix[sensor]
        for idx, reading in enumerate(imu_readings):
            gyr = r_level @ reading.raw_gyr
            acc = r_level @ reading.raw_acc
            mag = r_level @ reading.raw_mag

            imu_readings[idx].acc = acc
            # imu_readings[idx].gyr = gyr
            imu_readings[idx].mag = mag

        oriented_magnetometer_readings = [imu.mag for imu in imu_readings]
        sensor_magnetic_correction[sensor] = calibrate_magnetometer(oriented_magnetometer_readings)
        print(sensor_magnetic_correction[sensor]['offset'])
        print(sensor_magnetic_correction[sensor]['scale_matrix'])

        print(f"Applying calibration for {sensor}")
        for idx, reading in enumerate(imu_readings):
            offset = np.array(sensor_magnetic_correction[sensor]['offset'])
            scale_matrix = np.array(sensor_magnetic_correction[sensor]['scale_matrix'])

            calibrated_magnetometer = (raw_vector := reading.mag - offset) @ scale_matrix.T
            imu_readings[idx].mag = calibrated_magnetometer

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