import json
from dataclasses import astuple
from typing import List

import numpy as np
from numpy import random

from data_tools import Database, IMUReading, estimate_level_correction, SENSOR_LIST, calibrate_magnetometer

db_file = "data/dataset.db"
calibration_file = "data/calibration.json"

calibration_data = {}

def main():
    db = Database(db_file)

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

    for sensor in SENSOR_LIST:
        calibration_data[sensor] = {}
        print(f"Solving orientation calibration for {sensor}")
        per_sensor_imu = [reading for reading in zero_speed_imu if reading.sensor_name == sensor]
        random_samples = random.choice(per_sensor_imu, size=int(0.75*len(per_sensor_imu)))
        calibration_data[sensor]['rotation_matrix'] = estimate_level_correction(random_samples, len(random_samples)).tolist()

    for sensor in SENSOR_LIST:
        print(f"Reading entries for magnetic calibration of {sensor}")
        sensor_readings = [imu for imu in zero_speed_imu if imu.sensor_name == sensor]

        print(f"Solving magnetic calibration for {sensor} using {len(sensor_readings)} readings")

        # Apply rotation matrix prior to magnetic calibration
        r_level = calibration_data[sensor]['rotation_matrix']

        oriented_magnetometer_readings = [r_level @ imu.raw_mag for imu in sensor_readings]
        offset, scale_matrix = calibrate_magnetometer(oriented_magnetometer_readings).values()

        calibration_data[sensor]['magnetic_offset'] = offset.tolist()
        calibration_data[sensor]['magnetic_scale_matrix'] = scale_matrix.tolist()

        print(f"Magnetic calibrations complete:")
        print(f"Offset: \n", offset)
        print(f"Scaling matrix: \n", scale_matrix)

    print(f"Saving calibration data...")

    with open(calibration_file, "w") as f:
        json.dump(calibration_data, f, indent=4)

            # print("Running batch calibration updates...")
        # def apply_calibrations(batch, current_idx, total_count):
        #
        #     print(f"\033[K{current_idx}/{total_count} ({100*(current_idx/total_count):.2f}%) -- {len(batch)} in batch", end="\r", flush=True)
        #
        #     modified_batch = []
        #     for sample in batch:
        #         # apply level corrections
        #         sample.acc = r_level @ sample.raw_acc
        #         sample.gyr = r_level @ sample.raw_gyr
        #         oriented_mag = r_level @ sample.raw_mag
        #
        #         sample.mag = (raw_vector := oriented_mag - offset) @ scale_matrix.T
        #
        #         modified_batch.append(sample)
        #
        #     db.write_imu(modified_batch)
        #
        # db.iterate_batches(
        #     table_name="imu",
        #     callback=apply_calibrations,
        #     batch_size=1000000
        # )


        # print(f"Applying calibration for {sensor}")
        # for idx, reading in enumerate(imu_readings):
        #     offset = np.array(sensor_magnetic_correction[sensor]['offset'])
        #     scale_matrix = np.array(sensor_magnetic_correction[sensor]['scale_matrix'])
        #
        #     calibrated_magnetometer = (raw_vector := reading.mag - offset) @ scale_matrix.T
        #     imu_readings[idx].mag = calibrated_magnetometer
        #
        #
        # print(f"Updating database for {sensor}")
        # batch_size = 10000
        # for i in range(0, len(imu_readings), batch_size):
        #     batch = imu_readings[i: i + batch_size]
        #     db.write_imu(batch)

if __name__ == '__main__':
    main()