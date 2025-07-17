import os
import pandas as pd
import s3fs
from datetime import datetime, timedelta, timezone
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-base-path", type=str, required=True, help="S3 path to event base directory")
    parser.add_argument("--install-file", type=str, required=True, help="S3 or local path to install times CSV")
    parser.add_argument("--feature-columns", type=str, required=True, help="Comma-separated list of feature columns")
    parser.add_argument("--output-dir", type=str, required=True, help="S3/local path to write output train_raw parquet")
    return parser.parse_args()

def main():
    args = parse_args()
    fs = s3fs.S3FileSystem(anon=False)

    # --- Step 1: Load install_time table ---
    install_df = pd.read_csv(args.install_file, storage_options={'anon': False} if args.install_file.startswith("s3://") else None)
    install_df["estimated_install_time"] = pd.to_datetime(install_df["estimated_install_time"])

    # --- Step 2: Select users installed 91-121 days ago ---
    today = datetime.now(timezone.utc).date()
    cutoff_min = today - timedelta(days=121)
    cutoff_max = today - timedelta(days=91)
    train_users_df = install_df[
        (install_df["estimated_install_time"].dt.date >= cutoff_min) &
        (install_df["estimated_install_time"].dt.date <= cutoff_max)
    ].copy()
    user_ids_for_training = set(train_users_df["user_id"])
    print(f"Users selected for training: {len(user_ids_for_training)}")
    if not user_ids_for_training:
        raise ValueError("No users found in the specified install window.")

    # --- Step 3: Scan ALL partitions for potential events (0–90 days after install for any user in cohort) ---
    min_install = train_users_df["estimated_install_time"].min().date()
    max_install = train_users_df["estimated_install_time"].max().date()
    scan_start = min_install
    scan_end   = max_install + timedelta(days=90)
    dates_to_scan = pd.date_range(scan_start, scan_end)
    partition_paths = [
        f"{args.event_base_path}/d={d.strftime('%Y-%m-%d')}" for d in dates_to_scan
    ]

    # --- Step 4: Read all .parquet event files for those partitions and filter user_ids ---
    FEATURE_COLUMNS = [col.strip() for col in args.feature_columns.split(",")]
    all_dfs = []
    for partition_path in partition_paths:
        try:
            files = fs.ls(partition_path)
        except Exception as e:
            print(f"Warning: could not list files for {partition_path}: {e}")
            continue
        parquet_files = [f for f in files if f.endswith(".parquet")]
        for file in parquet_files:
            try:
                df = pd.read_parquet(f"s3://{file}", columns=FEATURE_COLUMNS, filesystem=fs)
                df = df[df["user_id"].isin(user_ids_for_training)]
                if not df.empty:
                    all_dfs.append(df)
            except Exception as e:
                print(f"Could not load {file}: {e}")

    if not all_dfs:
        raise ValueError("No events found for selected users in partitions.")

    event_df = pd.concat(all_dfs, ignore_index=True)
    print("Loaded event data:", event_df.shape)

    # --- Step 5: Attach install_time to each event, calculate days_after_install, filter only [0, 90] days ---
    event_df = event_df.merge(
        train_users_df[["user_id", "estimated_install_time"]],
        on="user_id", how="left"
    )
    event_df["event_time"] = pd.to_datetime(event_df["event_time"], errors="coerce")
    event_df["days_after_install"] = (
        event_df["event_time"] - event_df["estimated_install_time"]
    ).dt.days

    event_df_0_90 = event_df[event_df["days_after_install"].between(0, 90)].copy()
    print(f"Events in first 0–90 days after install: {event_df_0_90.shape}")

    # --- Step 6: Save full cohort (0–90 days of events) to S3/local for feature engineering ---
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = os.path.join(args.output_dir, run_id, "train_raw.parquet")
    if output_path.startswith("s3://"):
        event_df_0_90.to_parquet(output_path, index=False, storage_options={'anon': False})
    else:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        event_df_0_90.to_parquet(output_path, index=False)
    print("Saved 0–90 day event data to:", output_path)

    print(event_df_0_90.head(10))

if __name__ == "__main__":
    main()