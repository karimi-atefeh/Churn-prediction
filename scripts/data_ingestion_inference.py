import os
import argparse
import pandas as pd
import s3fs
from datetime import datetime, timedelta, timezone

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events-dir",      type=str, required=True)
    parser.add_argument("--install-file",    type=str, required=True)
    parser.add_argument("--output-dir",      type=str, required=True)
    parser.add_argument("--feature-columns", type=str, default=None)
    parser.add_argument("--lookback-days",   type=int, default=20)
    parser.add_argument("--target-days-ago", type=int, default=8)
    return parser.parse_args()

def extract_date(path):
    for part in path.split("/"):
        if part.startswith("d="):
            return datetime.strptime(part.split("=")[-1], "%Y-%m-%d").date()
    return None

def main():
    args = parse_args()
    print("[DEBUG] EVENTS DIR =", args.events_dir)
    fs = s3fs.S3FileSystem(anon=False)

    # 1. Read or update install times
    if args.install_file.startswith("s3://"):
        install_df = pd.read_csv(
            args.install_file,
            storage_options={"anon": False}
        )
    else:
        install_df = pd.read_csv(args.install_file)

    install_df["estimated_install_time"] = pd.to_datetime(
    install_df["estimated_install_time"]
)

    # 2. List recent event folders
    today  = datetime.now(timezone.utc).date()
    cutoff = today - timedelta(days=args.lookback_days)
    all_paths = fs.ls(args.events_dir)
    recent = [p for p in all_paths if "/d=" in p and extract_date(p) >= cutoff]

    # 3. Load those Parquets
    df_list = []
    for folder in recent:
        for f in fs.ls(folder):
            if f.endswith(".parquet"):
                df_list.append(
                    pd.read_parquet(f"s3://{f}", columns=(args.feature_columns.split(",") if args.feature_columns else None),
                                    storage_options={"anon": False})
                )
    if not df_list:
        raise ValueError("No event data loaded.")
    raw_df = pd.concat(df_list, ignore_index=True)
    raw_df["event_time"] = pd.to_datetime(raw_df["event_time"], errors="coerce")

    # 4. Add any brand-new installs
    existing = set(install_df["user_id"])
    users    = set(raw_df["user_id"].unique())
    new_u    = users - existing
    if new_u:
        mins = (
            raw_df[raw_df["user_id"].isin(new_u)]
              .groupby("user_id")["event_time"].min()
              .reset_index(name="estimated_install_time")
        )
        install_df = pd.concat([install_df, mins], ignore_index=True)

        if args.install_file.startswith("s3://"):
            # Save directly back to the original S3 file (overwrite)
            install_df.to_csv(args.install_file, index=False, storage_options={"anon": False})
            print("Updated install CSV →", args.install_file)
        else:
            install_df.to_csv(args.install_file, index=False)
            print("Updated install CSV →", args.install_file)

    # 5. Filter users who installed exactly N days ago
    target = today - timedelta(days=args.target_days_ago)
    tui = install_df[install_df["estimated_install_time"].dt.date == target]["user_id"]
    if tui.empty:
        raise ValueError("No users installed on target day")
    final_df = raw_df[raw_df["user_id"].isin(tui)].copy()
    print("Final shape:", final_df.shape)

    # 6. Write inference-raw
    run_id    = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    out_path  = f"{args.output_dir.rstrip('/')}/{run_id}/inference_raw.parquet"
    if out_path.startswith("s3://"):
        final_df.to_parquet(out_path, index=False, storage_options={"anon": False})
    else:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        final_df.to_parquet(out_path, index=False)
    print("Saved inference data →", out_path)

if __name__ == "__main__":
    main()
