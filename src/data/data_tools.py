import csv
import io
import re
from dataclasses import dataclass, field, astuple
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timezone
import traceback
import sqlite3

import numpy as np
from pyproj import Proj

from scipy.spatial.transform import Rotation


@dataclass
class IMUReading:
    gps_second: float
    gps_time_elapsed: int
    sensor_name: str
    sensor_type: str
    sensor_id: int
    source_time: int
    raw_gyr: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    raw_acc: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    raw_mag: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    raw_tmp: float = 0
    gyr: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    acc: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    mag: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    tmp: float = 0
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


def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)

class Database:
    def __init__(self, db_path: str):
        # Create a database and tables if they don't exist
        try:

            self.conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
            self.cursor = self.conn.cursor()

            create_imu_table = '''
                CREATE TABLE IF NOT EXISTS imu (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL UNIQUE,
                    source_time INTEGER NOT NULL,
                    gps_second INTEGER NOT NULL,
                    gps_time_elapsed INTEGER NOT NULL,
                    sensor_name TEXT NOT NULL,
                    sensor_type TEXT NOT NULL,
                    sensor_id INTEGER NOT NULL,
                    raw_data array,
                    cal_data array
                );
            '''
            self.cursor.execute(create_imu_table)

            create_gps_table = '''
                CREATE TABLE IF NOT EXISTS gps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL UNIQUE,
                    latitude DECIMAL NOT NULL,
                    longitude DECIMAL NOT NULL,
                    altitude DECIMAL NOT NULL,
                    fix_indicator INTEGER NOT NULL,
                    satellite_count INTEGER NOT NULL,
                    geoid_separation DECIMAL NOT NULL,
                    pdop DECIMAL NOT NULL,
                    hdop DECIMAL NOT NULL,
                    vdop DECIMAL NOT NULL,
                    course DECIMAL NOT NULL,
                    speed DECIMAL NOT NULL,
                    utc_time DECIMAL NOT NULL,
                    date TEXT NOT NULL
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
                INSERT INTO gps (timestamp, latitude, longitude, altitude, fix_indicator, satellite_count, geoid_separation, pdop, hdop, vdop, course, speed, utc_time, date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(timestamp) DO UPDATE SET
                    latitude = excluded.latitude,
                    longitude = excluded.longitude,
                    altitude = excluded.altitude,
                    fix_indicator = excluded.fix_indicator,
                    satellite_count = excluded.satellite_count,
                    geoid_separation = excluded.geoid_separation,
                    pdop = excluded.pdop,
                    hdop = excluded.hdop,
                    vdop = excluded.vdop,
                    course = excluded.course,
                    speed = excluded.speed,
                    utc_time = excluded.utc_time,
                    date = excluded.date
            """

            # If entry is a single reading, place in list
            if isinstance(entry, GPSReading):
                entry = [entry]

            updates = [(
                value.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"),
                value.latitude,
                value.longitude,
                value.altitude,
                value.fix_indicator,
                value.satellite_count,
                value.geoid_separation,
                value.pdop,
                value.hdop,
                value.vdop,
                value.course,
                value.speed,
                value.utc_time,
                value.date
            ) for value in entry]
            self.cursor.executemany(query, updates)

            self.conn.commit()
        except Exception as e:
            traceback.print_exc()
            print("Failed to write gps entry to database")

    def write_imu(self, entry: IMUReading | List[IMUReading]):
        try:
            query ="""
                INSERT INTO imu (timestamp, source_time, gps_second, gps_time_elapsed, sensor_name, sensor_type, sensor_id, raw_data, cal_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(timestamp) DO UPDATE SET
                    source_time = excluded.source_time,
                    gps_second = excluded.gps_second,
                    gps_time_elapsed = excluded.gps_time_elapsed,
                    sensor_name = excluded.sensor_name,
                    sensor_type = excluded.sensor_type,
                    sensor_id = excluded.sensor_id,
                    raw_data = excluded.raw_data,
                    cal_data = excluded.cal_data
            """

            if isinstance(entry, IMUReading):
                entry = [entry]


            updates = [(value.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"),
                        value.source_time,
                        value.gps_second,
                        value.gps_time_elapsed,
                        value.sensor_name,
                        value.sensor_type,
                        value.sensor_id,
                        np.concatenate([value.raw_gyr, value.raw_acc, value.raw_mag, np.array([value.raw_tmp])]),
                        np.concatenate([value.gyr, value.acc, value.mag, np.array([value.tmp])])
                        ) for value in entry]
            self.cursor.executemany(query, updates)

            self.conn.commit()
        except Exception as e:
            traceback.print_exc()
            print("Failed to write imu entry to database")

    def to_gps(self, query) -> GPSReading | List[GPSReading]:
        """
        Converts the results of a GPS query to a single GPSReading or a List of GPSReadings.
        """
        rows = query.fetchall()
        if not rows:
            return []

        results = []
        for row in rows:
            # Mapping based on the CREATE TABLE gps structure:
            # 0:timestamp, 1:latitude, 2:longitude, 3:altitude, 4:fix_indicator,
            # 5:satellite_count, 6:geoid_separation, 7:pdop, 8:hdop, 9:vdop,
            # 10:course, 11:speed, 12:utc_time, 13:date
            reading = GPSReading(
                timestamp=datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S.%f"),
                latitude=float(row[1]),
                longitude=float(row[2]),
                altitude=float(row[3]),
                fix_indicator=int(row[4]),
                satellite_count=int(row[5]),
                geoid_separation=float(row[6]),
                pdop=float(row[7]),
                hdop=float(row[8]),
                vdop=float(row[9]),
                course=float(row[10]),
                speed=float(row[11]),
                utc_time=float(row[12]),
                date=row[13]
            )
            results.append(reading)

        return results[0] if len(array_results := results) == 1 else results


    def to_imu(self, query) -> IMUReading | List[IMUReading]:
        """
        Converts the results of an IMU query to a single IMUReading or a List of IMUReadings.
        Reconstructs the concatenated arrays back into gyr, acc, mag, and tmp components.
        """
        rows = query.fetchall()
        if not rows:
            return []

        results = []
        for row in rows:
            # Mapping based on the CREATE TABLE imu structure:
            # 0:timestamp, 1:source_time, 2:sensor_name, 3:sensor_type,
            # 4:sensor_id, 5:raw_data (array), 6:cal_data (array)
            ts = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S.%f")
            src_time = row[1]
            gps_sec = row[2]
            gps_elapsed = row[3]
            name = row[4]
            s_type = row[5]
            s_id = row[6]
            raw_arr = row[7]  # Automatically converted back to np.ndarray via converter
            cal_arr = row[8]  # Automatically converted back to np.ndarray via converter

            # Helper to unpack the 10-element array [gyr(3), acc(3), mag(3), tmp(1)]
            def unpack_array(arr):
                return (
                    arr[0:3],   # gyr
                    arr[3:6],   # acc
                    arr[6:9],   # mag
                    float(arr[9]) # tmp
                )

            raw_gyr, raw_acc, raw_mag, raw_tmp = unpack_array(raw_arr)
            cal_gyr, cal_acc, cal_mag, cal_tmp = unpack_array(cal_arr)

            reading = IMUReading(
                timestamp=ts,
                source_time=src_time,
                gps_second=gps_sec,
                gps_time_elapsed=gps_elapsed,
                sensor_name=name,
                sensor_type=s_type,
                sensor_id=s_id,
                raw_gyr=raw_gyr,
                raw_acc=raw_acc,
                raw_mag=raw_mag,
                raw_tmp=raw_tmp,
                gyr=cal_gyr,
                acc=cal_acc,
                mag=cal_mag,
                tmp=cal_tmp
            )
            results.append(reading)

        return results[0] if len(results) == 1 else results

    def _format_dt(self, dt: Optional[datetime]) -> Optional[str]:
        """Helper to convert datetime objects to the database string format."""
        if dt is None:
            return None
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f")

    def get_imu_readings(
            self,
            sensor_name: Optional[str] = None,
            sensor_type: Optional[str] = None,
            sensor_id: Optional[int] = None,
            start_datetime: Optional[datetime] = None,
            end_datetime: Optional[datetime] = None
    ) -> IMUReading | List[IMUReading]:
        """
        Retrieves IMU readings from the database with optional filters.
        """
        query = """
            SELECT 
                timestamp, source_time, gps_second, gps_time_elapsed, sensor_name, sensor_type, 
                sensor_id, raw_data, cal_data 
            FROM imu
        """
        conditions = []
        params = []

        if sensor_name:
            conditions.append("sensor_name = ?")
            params.append(sensor_name)

        if sensor_type:
            conditions.append("sensor_type = ?")
            params.append(sensor_type)

        if sensor_id is not None:
            conditions.append("sensor_id = ?")
            params.append(sensor_id)

        if start_datetime:
            conditions.append("timestamp >= ?")
            params.append(self._format_dt(start_datetime))

        if end_datetime:
            conditions.append("timestamp <= ?")
            params.append(self._format_dt(end_datetime))

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        try:
            self.cursor.execute(query, params)
            return self.to_imu(self.cursor)
        except Exception as e:
            traceback.print_exc()
            print(f"Error querying IMU data: {e}")
            return []

    def get_gps_readings(
            self,
            start_datetime: Optional[datetime] = None,
            end_datetime: Optional[datetime] = None,
            min_latitude: Optional[float] = None,
            max_latitude: Optional[float] = None,
            min_longitude: Optional[float] = None,
            max_longitude: Optional[float] = None,
            min_speed: Optional[float] = None,
            max_speed: Optional[float] = None,
            min_course: Optional[float] = None,
            max_course: Optional[float] = None,
            fix_indicator: Optional[int] = None,
            min_satellite_count: Optional[int] = None
    ) -> GPSReading | List[GPSReading]:
        """
        Retrieves GPS readings from the database with optional filters.
        """
        query = """
            SELECT 
                timestamp, latitude, longitude, altitude, fix_indicator, 
                satellite_count, geoid_separation, pdop, hdop, vdop, 
                course, speed, utc_time, date 
            FROM gps
        """
        conditions = []
        params = []

        # Time filters
        if start_datetime:
            conditions.append("timestamp >= ?")
            params.append(self._format_dt(start_datetime))
        if end_datetime:
            conditions.append("timestamp <= ?")
            params.append(self._format_dt(end_datetime))

        # Coordinate filters
        if min_latitude is not None:
            conditions.append("latitude >= ?")
            params.append(min_latitude)
        if max_latitude is not None:
            conditions.append("latitude <= ?")
            params.append(max_latitude)
        if min_longitude is not None:
            conditions.append("longitude >= ?")
            params.append(min_longitude)
        if max_longitude is not None:
            conditions.append("longitude <= ?")
            params.append(max_longitude)

        # Speed/Course filters
        if min_speed is not None:
            conditions.append("speed >= ?")
            params.append(min_speed)
        if max_speed is not None:
            conditions.append("speed <= ?")
            params.append(max_speed)
        if min_course is not None:
            conditions.append("course >= ?")
            params.append(min_course)
        if max_course is not None:
            conditions.append("course <= ?")
            params.append(max_course)

        # Hardware/Fix filters
        if fix_indicator is not None:
            conditions.append("fix_indicator = ?")
            params.append(fix_indicator)
        if min_satellite_count is not None:
            conditions.append("satellite_count >= ?")
            params.append(min_satellite_count)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        try:
            self.cursor.execute(query, params)
            return self.to_gps(self.cursor)
        except Exception as e:
            traceback.print_exc()
            print(f"Error querying GPS data: {e}")
            return []

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
                        gps_second=float(row.get('gps_second') or 0),
                        gps_time_elapsed=int(row.get('gps_time_elapsed') or 0),
                        sensor_name=row.get('sensor_id'),
                        sensor_type="_".join(row.get('sensor_id').split('_')[:-1]),
                        sensor_id=int(row.get('sensor_id').split('_')[-1]),
                        source_time=int(row.get('timestamp') or 0),
                        raw_acc=np.array([float(row.get('accel_x') or 0), float(row.get('accel_y') or 0), float(row.get('accel_z') or 0)]),
                        raw_gyr=np.array([float(row.get('gyro_x') or 0), float(row.get('gyro_y') or 0), float(row.get('gyro_z') or 0)]),
                        raw_mag=np.array([float(row.get('mag_x') or 0), float(row.get('mag_y') or 0), float(row.get('mag_z') or 0)]),
                        raw_tmp=float(row.get('temp') or 0)
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
                return {
                    "utc_time": data[1],
                    "fix_indicator": fix_indicator,
                    "latitude": 0,
                    "longitude": 0,
                    "altitude": 0,
                    "satellite_count": int(data[7] or 0),
                    "geoid_separation": float(data[11] or 0),
                }

            return {
                "utc_time": float(data[1] or 0),
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
                "pdop": float(data[-4] or 0),
                "hdop": float(data[-3] or 0),
                "vdop": float(data[-2] or 0),
            }
        # parse velocity data
        elif "VTG" in line:
            return {
                "course": float(data[1] or 0),
                "speed": float(data[7] or 0),
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

                        if reading.timestamp > datetime(2020, 1, 1):
                            # add reading to list of entries
                            gps_entries.append(reading)

                    except Exception as e:
                        print(e)

                # create a new entry using the GGA line
                gps_data = parse_gps(line)

                # if the fix is not valid, de-initialize the gps_data
                # if gps_data.get("fix_indicator",0) == 0:
                #     gps_data = None

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