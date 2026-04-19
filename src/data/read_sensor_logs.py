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

        # align GPS and IMU entries to establish timestamps
        for index, imu_entry in enumerate(imu_entries):
            if float(imu_entry.gps_second) in time_hash:
                imu_entry.timestamp = time_hash.get(float(imu_entry.gps_second)) + timedelta(microseconds=int(imu_entry.gps_time_elapsed))
            else:
                imu_entry.timestamp = None

        # some imu readings were taken before valid fix and may not have a matching gps entry
        # filter entries that do not have valid matches
        imu_entries = [entry for entry in imu_entries if entry.timestamp is not None]

        # sort entries in order of source time
        imu_entries = sorted(imu_entries, key=lambda entry: entry.source_time)


        time_modifier = 0
        start_index = None

        # loop over entries to identify gaps and correct write buffer errors
        for index, imu_entry in enumerate(imu_entries):
            if index == 0:
                continue

            dt = (imu_entry.timestamp - imu_entries[index-1].timestamp).total_seconds()

            if start_index is not None and index - start_index > 600:
                start_index = None
                time_modifier = 0
                continue

            if dt < -.01 and time_modifier == 0:
                print(f"\tFound negative delta time at {index} -- {dt} seconds lost starting at {imu_entry.timestamp}")
                start_index = index
                time_modifier = -1 * dt

            if dt > 0.05 and time_modifier > 0.01:
                gap_start_time = imu_entries[start_index-1].timestamp
                gap_end_time = imu_entries[index].timestamp
                gap_duration = gap_end_time - gap_start_time

                fill_start_time = imu_entries[start_index].timestamp
                fill_end_time = imu_entries[index-1].timestamp
                fill_duration = fill_end_time - fill_start_time

                print(f"\t -- Filling {gap_duration.total_seconds()} second gap with {fill_duration.total_seconds()} seconds of data.")

                time_modifier += (gap_duration - fill_duration).total_seconds() / 2
                # time_modifier = (dt + time_modifier) / 2 if dt > time_modifier else time_modifier
                # time_modifier += dt / 2
                print(f"\t -- Shifting samples {start_index} to {index} ({index - start_index} samples). -> {time_modifier} sec")
                print(f"\t -- {imu_entries[start_index].timestamp} - {imu_entries[index].timestamp}")
                for sub_index in range(start_index, index):
                    imu_entries[sub_index].timestamp += timedelta(seconds=time_modifier)
                time_modifier = 0
                start_index = None

        # for index, imu_entry in enumerate(imu_entries):
        #
        #     if index % 1000 == 0:
        #         print(f"\tAligning timestamps {index}/{len(imu_entries)} -- {100 * (index/len(imu_entries)):.2f}%", end='\r', flush=True)
        #
        #     # skip entries that don't match a gps entry
        #     if not float(imu_entry.gps_second) in time_hash:
        #         imu_entry.timestamp = None
        #         continue
        #
        #     # Check if this is the first in the series
        #     if not imu_entry.gps_second == prev_gps_second:
        #
        #         # If previous entries were modified we need to shift them by half the delta between the first in the new second
        #         if time_modifier > 0 and not imu_entries[index - 1].timestamp is None:
        #             timestamp = time_hash.get(imu_entry.gps_second) + timedelta(microseconds=int(imu_entry.gps_time_elapsed))
        #             dt = (timestamp - imu_entries[index - 1].timestamp).total_seconds()
        #
        #             for i in range(gps_second_start_index, index):
        #                 if imu_entries[i].gps_second == prev_gps_second:
        #                     imu_entries[i].timestamp += timedelta(seconds=float(dt / 2))
        #
        #             time_modifier = 0
        #             gps_second_start_index = index
        #
        #         prev_gps_second = imu_entry.gps_second
        #
        #     imu_entry.timestamp = time_hash.get(imu_entry.gps_second)
        #     imu_entry.timestamp += timedelta(microseconds=int(imu_entry.gps_time_elapsed))
        #     imu_entry.timestamp += timedelta(seconds=int(time_modifier))
        #
        #     if index == 0 or imu_entries[index - 1].timestamp is None:
        #         continue
        #
        #     # calculate the time elapsed between samples
        #     dt = (imu_entry.timestamp - imu_entries[index - 1].timestamp).total_seconds()
        #
        #     # if time is negative roll delta into this second
        #     if dt < 0:
        #         time_modifier = -1 * dt
        #         imu_entry.timestamp += timedelta(seconds=int(time_modifier))


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