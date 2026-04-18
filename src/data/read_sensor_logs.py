import argparse
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from data_tools import Database, read_imu_log, read_gps_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', type=Path)
    parser.add_argument('--db-file', type=Path)
    parser.add_argument('--check-db', type=bool, default=False)
    args = parser.parse_args()

    total_gps_entries = 0

    db = Database("../../data/dataset.db" if args.db_file is None else args.db_file)

    for gps_log in sorted(args.log_dir.glob('*gps.log')):
        timestamp = "_".join(gps_log.name.split("_")[0:2])
        imu_log = args.log_dir.joinpath(f'{timestamp}_imu.log')

        print(f"Reading file pair {gps_log.name} and {imu_log.name}")

        imu_entries = read_imu_log(imu_log)
        print(f"\tIMU entries: {len(imu_entries)}")

        gps_entries = read_gps_log(gps_log)
        print(f"\tGPS entries: {len(gps_entries)}")


        # store every gps entry time in a hash along with its timestamp
        time_hash = {}
        for gps_entry in gps_entries:
            time_hash[float(gps_entry.utc_time)] = gps_entry.timestamp

        time_modifier = 0
        prev_gps_second = 0
        is_first = True
        share_previous_gap = False

        for index, imu_entry in enumerate(imu_entries):

            if index % 1000 == 0:
                print(f"\tAligning timestamps {index}/{len(imu_entries)} -- {100 * (index/len(imu_entries)):.2f}%", end='\r', flush=True)

            # skip entries that don't match a gps entry
            if not float(imu_entry.gps_second) in time_hash:
                imu_entry.timestamp = None
                continue

            # Check if this is the first in the series
            if not imu_entry.gps_second == prev_gps_second:

                # If previous entries were modified we need to shift them by half the delta between the first in the new second
                if time_modifier > 0:
                    timestamp = time_hash.get(imu_entry.gps_second) + timedelta(microseconds=int(imu_entry.gps_time_elapsed))
                    dt = (timestamp - imu_entries[index - 1].timestamp).total_seconds()

                    for matched_imu_entry in imu_entries:
                        if matched_imu_entry.gps_second == prev_gps_second:
                            matched_imu_entry.timestamp += timedelta(seconds=float(dt / 2))

                    time_modifier = 0

                prev_gps_second = imu_entry.gps_second

            imu_entry.timestamp = time_hash.get(imu_entry.gps_second)
            imu_entry.timestamp += timedelta(microseconds=int(imu_entry.gps_time_elapsed))
            imu_entry.timestamp += timedelta(seconds=int(time_modifier))

            if index == 0 or imu_entries[index - 1].timestamp is None:
                continue

            # calculate the time elapsed between samples
            dt = (imu_entry.timestamp - imu_entries[index - 1].timestamp).total_seconds()

            # if time is negative roll delta into this second
            if dt < 0:
                time_modifier = -1 * dt
                imu_entry.timestamp += timedelta(seconds=int(time_modifier))


        # some imu readings were taken before valid fix and may not have a matching gps entry
        # filter entries that do not have valid matches
        imu_entries = [entry for entry in imu_entries if entry.timestamp is not None]

        total_gps_entries += len(gps_entries)
        print(f"\033[K", flush=True)

        print(f"\t{len(imu_entries)} matched across {len(gps_entries)} GPS entries."
              f"\n\tAverage IMU rate {len(imu_entries) / len(gps_entries)}"
              f"\n\t{len(gps_entries) / 60 :.2f} minutes this entry. {total_gps_entries / 60 :.2f} minutes total."
              f"\n\tStarting Time: {gps_entries[0].timestamp}"
              f"\n\tEnding Time: {imu_entries[-1].timestamp}", flush=True)

        if args.check_db:
            gps_entry = db.read_gps(time=gps_entries[-1].timestamp)
            imu_entry = db.read_imu(time=imu_entries[0].timestamp)
            print(f"\t\tHas IMU Entry: {imu_entry is not None}")
            print(f"\t\tHas GPS Entry: {gps_entry is not None}")
            if gps_entry and imu_entry:
                print("\t\tDatabase entries found. Skipping.")
                continue
            print("\t\tDatabase entries not found. Adding to database.")

        db.write_gps(gps_entries)
        db.write_imu(imu_entries)