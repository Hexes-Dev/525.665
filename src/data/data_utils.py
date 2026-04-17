import argparse
import csv
import pickle
import re
from dataclasses import dataclass, field, asdict, astuple
from pathlib import Path
from typing import List
from datetime import datetime, timezone
import traceback
import sqlite3

import numpy as np
from pyproj import Proj, Transformer

from scipy.spatial.transform import Rotation
from torch.utils.hipify.hipify_python import value
from torchgen.packaged.autograd.gen_trace_type import SELECT


@dataclass
class XYZ:
    x: float = 0
    y: float = 0
    z: float = 0

@dataclass
class IMUReading:
    gps_second: float
    gps_time_elapsed: int
    sensor_name: str
    sensor_type: str
    sensor_id: int
    raw_gyro: XYZ
    raw_accelerometer: XYZ
    raw_magnetometer: XYZ
    raw_temperature: float
    gyro: XYZ = field(default_factory=XYZ)
    accelerometer: XYZ = field(default_factory=XYZ)
    magnetometer: XYZ = field(default_factory=XYZ)
    temperature: float = 0
    timestamp: datetime = datetime(1970, 1, 1)

@dataclass
class GPSReading:
    latitude: float
    longitude: float
    altitude: float
    fix_indicator: int
    satellite_count: int
    geoid_separation: float
    pdop: float
    hdop: float
    vdop: float
    course: float
    speed: float
    utc_time: float
    date: str
    timestamp: datetime = datetime(1970, 1, 1)

SENSOR_LIST = [
    # "icm_20948_1",
    # "icm_20948_2",
    "lsm6dsox_3",
    "lsm6dsox_4",
    "lsm6dsox_5",
    "lsm6dsox_6",
]

def xyz_array(xyz: XYZ) -> np.ndarray:
    return np.array([
        xyz.x,
        xyz.y,
        xyz.z,
    ])

def array_xyz(np_array: np.ndarray) -> XYZ:
    return XYZ(
        x=float(np_array[0]),
        y=float(np_array[1]),
        z=float(np_array[2]),
    )

class Database:
    def __init__(self, db_path: str):
        # Create a database and tables if they don't exist
        try:
            self.conn = sqlite3.connect(db_path)
            self.cursor = self.conn.cursor()

            create_imu_table = '''
                CREATE TABLE IF NOT EXISTS imu (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL UNIQUE,
                    sensor_name TEXT NOT NULL,
                    sensor_type TEXT NOT NULL,
                    sensor_id INTEGER NOT NULL,
                    data BLOB NOT NULL
                );
            '''
            self.cursor.execute(create_imu_table)

            create_gps_table = '''
                CREATE TABLE IF NOT EXISTS gps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL UNIQUE,
                    latitude DECIMAL NOT NULL,
                    longitude DECIMAL NOT NULL,
                    data BLOB NOT NULL
                );
            '''
            self.cursor.execute(create_gps_table)
            self.conn.commit()

        except Exception as e:
            traceback.print_exc()
            print("Error creating database")

    def __del__(self):
        self.cursor.close()
        self.conn.close()

    def write_gps(self, entry: GPSReading | List[GPSReading]):
        try:
            query ="""
                INSERT INTO gps (timestamp, latitude, longitude, data)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(timestamp) DO UPDATE SET
                    latitude = excluded.latitude,
                    longitude = excluded.longitude,
                    data = excluded.data
            """

            if isinstance(entry, GPSReading):
                data = pickle.dumps(asdict(entry))
                self.cursor.execute(query, (
                    entry.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"),
                    entry.latitude,
                    entry.longitude,
                    data
                ))
            elif isinstance(entry, list):
                updates = [(value.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"),
                            value.latitude,
                            value.longitude,
                            pickle.dumps(asdict(value))) for value in entry]
                self.cursor.executemany(query, updates)

            self.conn.commit()
        except Exception as e:
            traceback.print_exc()
            print("Failed to write gps entry to database")

    def write_imu(self, entry: IMUReading | List[IMUReading]):
        try:
            query ="""
                INSERT INTO imu (timestamp, sensor_name, sensor_type, sensor_id, data)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(timestamp) DO UPDATE SET
                    sensor_name = excluded.sensor_name,
                    sensor_type = excluded.sensor_type,
                    sensor_id = excluded.sensor_id,
                    data = excluded.data
            """

            if isinstance(entry, IMUReading):
                data = pickle.dumps(asdict(entry))
                self.cursor.execute(query, (
                    entry.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"),
                    entry.sensor_name,
                    entry.sensor_type,
                    entry.sensor_id,
                    data
                ))

            elif isinstance(entry, list):
                updates = [(value.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"),
                            value.sensor_name,
                            value.sensor_type,
                            value.sensor_id,
                            pickle.dumps(asdict(value))) for value in entry]
                self.cursor.executemany(query, updates)

            self.conn.commit()
        except Exception as e:
            traceback.print_exc()
            print("Failed to write imu entry to database")


    def read_by_timestamp(self, table, start_time: datetime | None = None, end_time: datetime | None = None, time: datetime | None = None):
        if start_time and end_time:
            query = f"SELECT * FROM {table} WHERE timestamp BETWEEN ? AND ?"
            params = (start_time.strftime("%Y-%m-%d %H:%M:%S.%f"), end_time.strftime("%Y-%m-%d %H:%M:%S.%f"))
            self.cursor.execute(query, params)
        elif start_time:
            query = f"SELECT * FROM {table} WHERE timestamp >= ?"
            params = start_time.strftime("%Y-%m-%d %H:%M:%S.%f")
            self.cursor.execute(query, [params])
        elif end_time:
            query = f"SELECT * FROM {table} WHERE timestamp <= ?"
            params = end_time.strftime("%Y-%m-%d %H:%M:%S.%f")
            self.cursor.execute(query, [params])
        elif time:
            query = f"SELECT * FROM {table} WHERE timestamp = ?"
            params = time.strftime("%Y-%m-%d %H:%M:%S.%f")
            self.cursor.execute(query, [params])
        else:
            query = f"SELECT * FROM {table}"
            self.cursor.execute(query)

        return self.cursor.fetchall()

    def read_gps(self, start_time: datetime | None = None, end_time: datetime | None = None, time: datetime | None = None) -> GPSReading | List[GPSReading] | None:
        results =  self.read_by_timestamp("gps", start_time=start_time, end_time=end_time, time=time)
        if len(results) == 0:
            return None

        results = [GPSReading(**pickle.loads(entry[-1])) for entry in results]
        if len(results) == 1:
            return results[0]

        return results



    def read_imu(self, sensor_name: str | None = None, limit: int | None = None, start_time: datetime | None = None, end_time: datetime | None = None, time: datetime | None = None) -> IMUReading | List[IMUReading] | None:
        if sensor_name is not None:
            query = f"SELECT * FROM imu WHERE sensor_name = ? {"" if limit is None else f"LIMIT ?" }"
            params = (sensor_name, limit) if limit else (sensor_name,)
            self.cursor.execute(query, params)
            results = self.cursor.fetchall()
        else:
            results = self.read_by_timestamp("imu", start_time=start_time, end_time=end_time, time=time)

        if len(results) == 0:
            return None


        # print(f"Processing {len(results)} entries")
        # results = [IMUReading(**pickle.loads(entry[-1])) for entry in results]

        # Pre-define the attributes to avoid repeated string lookups in the loop
        attrs = (
            'raw_gyro', 'raw_magnetometer', 'raw_accelerometer',
            'gyro', 'magnetometer', 'accelerometer'
        )

        def transform_item(obj):
            """Processes a single object; returns None if it fails."""
            obj = IMUReading(**pickle.loads(obj[-1]))
            try:
                for attr in attrs:
                    # Direct access is faster than getattr/setattr in tight loops
                    # We use a generator expression to avoid creating intermediate lists
                    data = getattr(obj, attr).values()
                    setattr(obj, attr, XYZ(*(float(v) for v in data)))
                return obj
            except Exception as e:
                print(f"Failed entry: {e}")
                return None

        # A single list comprehension is significantly faster than a for-loop with .append()
        # We filter out 'None' values (the failed entries) in the same pass
        return [processed for processed in (transform_item(entry) for entry in results) if processed is not None]


"""
Reads entries from an IMU log and returns a list of IMUReadings
readings will be initialized with a default timestamp since IMU logs don't contain a date.
"""
# Read the entries from an IMU log into a list of IMUReading
def read_imu_log(path: Path) -> List[IMUReading]:

    imu_entries: List[IMUReading] = []

    # read the file in csv format to a dict
    with open(path, 'r') as imu_file:
        reader = csv.DictReader(imu_file)
        try:
            for row in reader:
                # map dict entries to an IMUReading object
                try:
                    reading = IMUReading(
                        gps_second=row['gps_second'],
                        gps_time_elapsed=row['gps_time_elapsed'],
                        sensor_name=row['sensor_id'],
                        sensor_type="_".join(row['sensor_id'].split('_')[:-1]),
                        sensor_id=int(row['sensor_id'].split('_')[-1]),
                        raw_accelerometer=XYZ(row.get('accel_x'), row.get('accel_y'), row.get('accel_z')),
                        raw_gyro=XYZ(row.get('gyro_x'), row.get('gyro_y'), row.get('gyro_z')),
                        raw_magnetometer=XYZ(row.get('mag_x'), row.get('mag_y'), row.get('mag_z')),
                        raw_temperature=float(row.get('temp') if row.get('temp') is not None else '0')
                    )

                    # add IMUReading to list of entries in file
                    imu_entries.append(reading)
                except Exception as e:
                    print(path)
                    print(row)
                    traceback.print_exc()
                    continue
        except Exception as e:
            print(path)
            traceback.print_exc()

        # return the list of all entries as IMUReadings
        return imu_entries

def parse_gps(line: str) -> dict:
    data = re.split(r'[,*]', line)
    try:
        # parse fix data
        if "GGA" in line:

            # return early if fix is invalid
            fix_indicator = int(data[6])
            if fix_indicator == 0:
                return {"fix_indicator": fix_indicator}

            return {
                "utc_time": data[1],
                "latitude": float(data[2]) if data[3] == "N" else ( - float(data[2])),
                "longitude": float(data[4]) if data[5] == "E" else ( - float(data[4])),
                "fix_indicator": int(data[6]),
                "satellite_count": int(data[7]),
                "altitude": float(data[9]),
                "geoid_separation": float(data[11]),
            }
        # parse satellite data
        elif "GSA" in line:
            return {
                "pdop": float(data[-4]),
                "hdop": float(data[-3]),
                "vdop": float(data[-2]),
            }
        # parse velocity data
        elif "VTG" in line:
            return {
                "course": float(data[1]),
                "speed": float(data[7]),
            }
        # parse date
        elif "RMC" in line:
            return {
                "date": data[9],
            }
    except Exception as e:
        traceback.print_exc()

    return {}


def read_gps_log(path: Path) -> List[GPSReading]:
    gps_entries: List[GPSReading] = []

    with open(path, 'r') as gps_file:

        gps_data = None

        for line in gps_file:

            # Check if the line is the start of a fix entry
            if line.startswith('$GNGGA'):

                # check that the previous reading is valid
                if gps_data is not None:
                    try:
                        # unpack the previous data into a reading
                        reading = GPSReading(**gps_data)

                        # update timestamp using data
                        timestring = f"{gps_data.get("utc_time")} {gps_data.get("date")}"
                        reading.timestamp = datetime.strptime(timestring, '%H%M%S.%f %d%m%y')
                        reading.timestamp.replace(tzinfo=timezone.utc)

                        # add reading to list of entries
                        gps_entries.append(reading)
                    except Exception as e:
                        print(e)

                # create a new entry using the GGA line
                gps_data = parse_gps(line)

                # if the fix is not valid, de-initialize the gps_data
                if gps_data.get("fix_indicator",0) == 0:
                    gps_data = None

            # check that the current fix is valid
            elif gps_data is not None:
                # union the current line with the fix data
                gps_data = gps_data | parse_gps(line)

    return gps_entries


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
    alt = -down  # NED down is negative altitude

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


def estimate_level_correction(imu_readings, n_samples=50):
    """
    Estimate a fixed rotation matrix to level the IMU
    using stationary samples where acc should equal [0, 0, 9.81].
    """
    acc_samples = []
    for r in imu_readings[:n_samples]:
        acc = np.array(astuple(r.raw_accelerometer), dtype=float)
        acc_samples.append(acc)

    # Mean acceleration vector while stationary
    acc_mean = np.mean(acc_samples, axis=0)
    acc_mean_norm = acc_mean / np.linalg.norm(acc_mean)

    # Target: gravity should point along +Z in NED
    target = np.array([0.0, 0.0, 1.0])

    # Find rotation from measured gravity direction to target
    # Using cross product to get rotation axis and angle
    axis = np.cross(acc_mean_norm, target)
    angle = np.arccos(np.clip(np.dot(acc_mean_norm, target), -1.0, 1.0))

    print(f"Leveling correction: {np.degrees(angle):.2f}° around axis {axis}")

    if np.linalg.norm(axis) < 1e-6:
        # Already level
        return Rotation.identity().as_matrix()

    axis = axis / np.linalg.norm(axis)
    R_level = Rotation.from_rotvec(axis * angle).as_matrix()
    return R_level

mag_cal = {
    'icm_20948_1': {
        'offset': [76.94760681, -66.96951594, 77.55489351],
        'scale_matrix': [[0.84153911, 0., 0.], [0., 1.25577818, 0.], [0., 0., 0.90268271]]
    },
    'icm_20948_2': {
        'offset': [-1.02907544, -42.41841608, 87.11279751],
        'scale_matrix': [[0.86475322, 0., 0.], [0., 1.38191788, 0.], [0., 0., 0.75332889]]
    },
    'lsm6dsox_3': {
        'offset': [17.20014811, -93.86711893, -67.32972263],
        'scale_matrix': [[0.92567556, 0., 0.], [0., 0.90971929, 0.], [0., 0., 1.16460516]]
    },
    'lsm6dsox_4': {
        'offset': [-7.14964275, -43.22469026, -95.31654668],
        'scale_matrix': [[0.90911492, 0., 0.], [0., 0.95921197, 0.], [0., 0., 1.13167312]]
    },
    'lsm6dsox_5': {
        'offset': [-39.82518789, -48.21792771, -51.20067354],
        'scale_matrix': [[1.05353745, 0., 0.], [0., 0.98599771, 0.], [0., 0., 0.96046484]]
    },
    'lsm6dsox_6': {
        'offset': [-22.8427984, -62.01316704, -63.64736925],
        'scale_matrix': [[1.09843453, 0., 0.], [0., 1.07129153, 0.], [0., 0., 0.83027393]]
    }
}

def calibrate_magnetometer(data):
    """
    Calibrates magnetometer for constrained motion (e.g., vehicle mounting).
    Uses Hard-Iron compensation and Axis-Aligned scaling.
    """
    data = np.array(data)
    if data.shape[0] < 2:
        raise ValueError("At least 2 points are required.")

    # 1. Hard-Iron Compensation (Offset)
    # Always calculate the mean to center the data.
    offset = np.mean(data, axis=0)
    centered_data = data - offset

    # 2. Axis-Aligned Soft-Iron Compensation (Scaling)
    # Instead of SVD, we calculate a scale factor for each axis individually.
    # We look at the standard deviation of each axis.
    stds = np.std(centered_data, axis=0)

    # We need a reference to know what "normal" scaling is.
    # We use the average standard deviation across all axes as our baseline.
    avg_std = np.mean(stds)

    # Avoid division by zero if the sensor hasn't moved at all
    if avg_std < 1e-9:
        scale_factors = np.array([1.0, 1.0, 1.0])
    else:
        # Scale factor is: (How much this axis moves) / (How much the average axis moves)
        # We use a threshold to prevent scaling axes that have almost no movement.
        scale_axis = stds / avg_std

        # If an axis has very low variance (like your Z-axis),
        # we don't want to scale it; we just leave it at 1.0.
        threshold = 0.1  # If movement is less than 10% of average, don't scale
        scale_factors = np.where(stds > (avg_std * threshold), scale_axis, 1.0)

    # The scale matrix is now a simple diagonal matrix.
    # This is much more stable for vehicle-mounted sensors.
    scale_matrix = np.diag(scale_factors)

    return {
        'offset': offset,
        'scale_matrix': scale_matrix
    }


def apply_magnetometer_calibration(reading: IMUReading, calibration: dict) -> IMUReading:
    """
    Applies the magnetometer calibration (offset and scale matrix)
    from a calibration dictionary to an IMUReading object.

    Args:
        reading: The IMUReading object to be updated.
        calibration: A dictionary containing 'offset' (ndarray)
                     and 'scale_matrix' (ndarray).

    Returns:
        The modified IMUReading object.
    """
    # 1. Extract the raw magnetometer values into a numpy array for math
    raw_vec = np.array([
        reading.raw_magnetometer.x,
        reading.raw_magnetometer.y,
        reading.raw_magnetometer.z
    ])

    # 2. Retrieve calibration parameters
    offset = np.array(calibration['offset'])
    scale_matrix = np.array(calibration['scale_matrix'])

    # 3. Apply the transformation: (Raw - Offset) @ ScaleMatrix.T
    # We use .T because in the math derivation, we are transforming a row vector
    calibrated_vec = (raw_vector := raw_vec - offset) @ scale_matrix.T

    # 4. Update the 'magnetometer' field with a new XYZ object
    # We cast to float() to ensure we store standard Python floats,
    # avoiding numpy-specific types inside our dataclass.
    reading.magnetometer = XYZ(
        x=float(calibrated_vec[0]),
        y=float(calibrated_vec[1]),
        z=float(calibrated_vec[2])
    )

    return reading


def export_imu_readings_to_csv(imu_readings: List['IMUReading'], filename: str):
    """
    Flattens and exports a list of IMUReading objects to a CSV file.

    Args:
        imu_readings: A list of IMUReading dataclass instances.
        filename: The destination path (e.g., 'data_export.csv').
    """
    if not imu_readings:
        print("No readings to export.")
        return

    # 1. Define the flattened header names
    # We expand the nested XYZ objects into individual columns for CSV compatibility
    headers = [
        'timestamp', 'gps_second', 'gps_time_elapsed', 'sensor_name',
        'sensor__type', 'sensor_id',
        'raw_gyro_x', 'raw_gyro_y', 'raw_gyro_z',
        'raw_accel_x', 'raw_accel_y', 'raw_accel_z',
        'raw_mag_x', 'raw_mag_y', 'raw_mag_z',
        'raw_temp',
        'gyro_x', 'gyro_y', 'gyro_z',
        'accel_x', 'accel_y', 'accel_z',
        'mag_x', 'mag_y', 'mag_z',
        'temp'
    ]

    try:
        with open(filename, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)

            for r in imu_readings:
                # 2. Flatten the object into a single list of values
                row = [
                    r.timestamp.isoformat(),  # Convert datetime to string
                    r.gps_second,
                    r.gps_time_elapsed,
                    r.sensor_name,
                    r.sensor_type,
                    r.sensor_id,
                    # Flattening raw_gyro
                    r.raw_gyro.x, r.raw_gyro.y, r.raw_gyro.z,
                    # Flattening raw_accelerometer
                    r.raw_accelerometer.x, r.raw_accelerometer.y, r.raw_accelerometer.z,
                    # Flattening raw_magnetometer
                    r.raw_magnetometer.x, r.raw_magnetometer.y, r.raw_magnetometer.z,
                    # Temperature
                    r.raw_temperature,
                    # Flattening processed gyro
                    r.gyro.x, r.gyro.y, r.gyro.z,
                    # Flattening processed accelerometer
                    r.accelerometer.x, r.accelerometer.y, r.accelerometer.z,
                    # Flattening processed magnetometer
                    r.magnetometer.x, r.magnetometer.y, r.magnetometer.z,
                    # Final temperature
                    r.temperature
                ]
                writer.writerow(row)

        print(f"Successfully exported {len(imu_readings)} records to '{filename}'.")

    except Exception as e:
        print(f"Failed to export CSV: {e}")