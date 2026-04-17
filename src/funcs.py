
import math
from typing import List

import ahrs
from pyproj import Proj
from scipy.spatial.transform import Rotation

import numpy as np
from scipy.spatial.transform import Rotation as R_func


def ned_to_latlon(position_ned: np.ndarray, start_lat: float, start_lon: float) -> tuple[float, float, float]:
    """
    Convert a single NED position offset (meters) to global lat/lon.

    Parameters
    ----------
    position_ned : np.ndarray
        Single position vector [north, east, down] in meters from origin.
    start_lat : float
        Starting latitude in decimal degrees.
    start_lon : float
        Starting longitude in decimal degrees.

    Returns
    -------
    tuple[float, float, float]
        (latitude, longitude, altitude) in decimal degrees and meters.
    """
    proj = Proj(f"+proj=aeqd +lat_0={start_lat} +lon_0={start_lon} +units=m")

    north = position_ned[0]
    east = position_ned[1]
    down = position_ned[2]

    lon, lat = proj(east, north, inverse=True)
    alt = -down  # Convert NED down to altitude

    return lat, lon, alt


def ddmm_to_decimal(ddmm: float) -> float:
    """
    Convert DDMM.MMMM format to decimal degrees.
    e.g. 3410.0982 → 34.168303
         -11912.1792 → -119.203200
    """
    sign = -1 if ddmm < 0 else 1
    ddmm = abs(ddmm)
    degrees = int(ddmm / 100)
    minutes = ddmm - (degrees * 100)
    return sign * (degrees + minutes / 60)


def apply_level_correction(v: np.ndarray, R_level: np.ndarray) -> np.ndarray:
    return R_level @ v


def estimate_level_correction(imu_readings, n_samples=100):
    """
    Estimate a fixed rotation matrix to level the IMU
    using stationary samples where acc should align with [0, 0, 1].
    """
    if not imu_readings or n_samples <= 0:
        return Rotation.identity().as_matrix()

    # Extract accelerometer data for a subset of readings
    subset = imu_readings[:n_samples]
    acc_data = np.array([
        [r.raw_accelerometer.x, r.raw_accelerometer.y, r.raw_accelerometer.z]
        for r in subset
    ], dtype=float)

    # Calculate the mean gravity vector
    acc_mean = np.mean(acc_data, axis=0)

    if np.linalg.norm(acc_mean) < 1e-6:
        return Rotation.identity().as_matrix()

    # Define target gravity vector [0, 0, 1]
    target = np.array([0.0, 0.0, 1.0])

    # Align vectors using Scipy
    try:
        rotation_obj, _ = Rotation.align_vectors([target], [acc_mean])
        R_level = rotation_obj.as_matrix()
        return R_level
    except Exception as e:
        print(f"Leveling estimation failed: {e}")
        return Rotation.identity().as_matrix()


def ekf_navigation(gyr: np.ndarray, acc: np.ndarray, mag: np.ndarray, time: np.ndarray, mag_ref : float | List[float] | None = None):
    """
    Estimates Attitude (Q), Velocity (V), and Position (P) using
    Attitude EKF + Integration.
    """
    number_of_samples = len(time)
    if number_of_samples == 0:
        return np.array([]), np.array([]), np.array([])

    # Pre-process: Ensure all arrays are (N, 3)
    gyr = gyr.reshape(len(gyr), 3)
    acc = acc.reshape(len(acc), 3)
    mag = mag.reshape(len(mag), 3)

    # Running EKF on data produced East / West mirror or GPS
    # This correction needs to be applied as part of calibration step
    gyr *= -1
    acc[:, [0, 1]] = acc[:, [1, 0]]
    mag[:,[0,1]] = mag[:,[1,0]]

    # Calculate accelerometer stationary bias using the first 100 samples
    acc_bias = np.mean(acc[:100], axis=0)


    # EKF can run on 6 dof or 9 dof data. An empy mag array is used to tell it we're using 9 dof data
    # Not currently using magnetic reference
    filter = ahrs.filters.EKF(
        frame="NED",
        mag=mag[0],
    )

    # Pre-allocate arrays
    Q = np.zeros((number_of_samples, 4))
    V = np.zeros((number_of_samples, 3))
    P = np.zeros((number_of_samples, 3))

    from ahrs.common.orientation import ecompass

    # Calculate initial orientation from stationary samples
    Q[0] = ecompass(np.mean(acc[:50], axis=0), np.mean(mag[:50], axis=0), frame="NED", representation='quaternion')

    for t in range(number_of_samples):

        # Normalize acceleration and magnetic readings
        acc_norm = acc[t] / np.linalg.norm(acc[t])
        mag_norm = mag[t] / np.linalg.norm(mag[t])

        # Handle first sample to establish dt
        if t == 0:
            Q[t] = filter.update(
                Q[0],
                gyr[t],
                acc_norm,
                mag_norm,
                dt=0
            )
            continue

        # Calculate time difference
        dt = (time[t] - time[t - 1]).total_seconds()

        if dt < 0:
            continue

        # Update Attitude EKF
        Q[t] = filter.update(
            q=Q[t - 1],
            gyr=gyr[t],
            acc=acc_norm,
            mag=mag_norm,
            dt=dt
        )

        # Convert Quaternion to Rotation Matrix (Sensor -> World)
        rot_matrix = R_func.from_quat(Q[t], scalar_first=True).as_matrix()

        # Transform acceleration to NED frame and remove bias
        acc_world = rot_matrix @ (acc[t] - acc_bias)

        # Integrate Velocity
        V[t] = V[t - 1] + (acc_world * dt)

        # Integrate position
        P[t] = P[t - 1] + V[t] * dt

        # Print update info
        if t % 1000 == 0 :
            print(f"\tEKF Checkpoint {t}")
            print(f"\t\tAcceleration Estimate: {acc_world}")
            print(f"\t\tVelocity Estimate: {V[t]}")
            mag_e = math.degrees(math.atan2(mag[t, 1], mag[t, 0])) % 360
            print(f"\t\tMagnetic Course Estimate: {mag_e}")
            vel_e = math.degrees(math.atan2(V[t, 1], V[t, 0])) % 360
            print(f"\t\tVelocity Course Estimate: {vel_e}")

    return Q, V, P


def ekf_to_coor(start_lat, start_lon, position):
    ekf_coor = []
    for p in position:
        lat, lon, alt = ned_to_latlon(p, start_lat, start_lon)
        ekf_coor.append((lat, lon, alt))

    return ekf_coor