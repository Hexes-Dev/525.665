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
            time_hash[gps_entry.utc_time] = gps_entry.timestamp

        samples_aligned = False
        max_count = 0

        for index, gps_second in enumerate(time_hash.keys()):
            samples = [imu for imu in imu_entries if float(imu.gps_second) == float(gps_second)]
            count = len(samples)
            max_count = max(count, max_count)
            print(f"\tScanning timestamps {index+1}/{len(time_hash.keys())} -- Stamp: {gps_second} -- Samples this stamp: {count} -- Max samples per stamp: {max_count}", end="\r", flush=True)
            if count > 650:
                anchor_imu = samples[0]
                print("\n\tAligning IMU samples using source time offset.")

                print(f"\t{anchor_imu.source_time}")
                offset_seconds = time_hash[gps_second].timestamp() - (anchor_imu.source_time / 1e6)
                for e in imu_entries:
                    e.timestamp = datetime.fromtimestamp(e.source_time / 1e6 + offset_seconds)

                samples_aligned = True
                break

        if not samples_aligned:
            print(f"\tAligning IMU samples via GPS second matching.")
            # If no anchor group is found, use the original relative alignment logic
            for index, imu_entry in enumerate(imu_entries):
                if imu_entry.gps_second in time_hash:
                    timestamp = time_hash.get(imu_entry.gps_second)
                    imu_entry.timestamp = timestamp + timedelta(microseconds=int(imu_entry.gps_time_elapsed))
                else:
                    imu_entries[index].timestamp = None

        # some imu readings were taken before valid fix and may not have a matching gps entry
        # filter entries that do not have valid matches
        imu_entries = [entry for entry in imu_entries if entry.timestamp is not None]

        total_gps_entries += len(gps_entries)

        print(f"\t{len(imu_entries)} matched across {len(gps_entries)} GPS entries."
              f"\n\tAverage IMU rate {len(imu_entries) / len(gps_entries)}"
              f"\n\t{len(gps_entries) / 60 :.2f} minutes this entry. {total_gps_entries / 60 :.2f} minutes total."
              f"\n\tStarting Time: {gps_entries[0].timestamp}"
              f"\n\tEnding Time: {imu_entries[-1].timestamp}")

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