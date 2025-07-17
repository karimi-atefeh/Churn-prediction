import os
import argparse
import pandas as pd
import s3fs
from datetime import datetime
import glob
import re

# --- Data Cleaning Function ---
def clean_events(df):
    """Basic data cleaning: drop missing, normalize, filter."""
    df = df.dropna(subset=["user_id", "event_time", "event_type", "session_id"])
    df["device_type"] = df["device_type"].fillna("unknown")
    df["country"] = df["country"].fillna("unknown")
    df["paying"] = df["paying"].fillna(False).astype(str).str.lower().eq("true")
    df = df.drop_duplicates()
    df = df[df["session_id"] != -1]
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df = df[(df["event_time"] >= "2018-01-01") & (df["event_time"] <= "2030-12-31")]
    return df

# --- Churn Labeling Function ---
def label_churn(df, install_df):
    """
    Assign churn=1 if NO events in days 30-90 after install, churn=0 otherwise.
    Retained (churn=0) means at least one event between day 30 and 90 after install.
    """
    installs = (
        df.groupby("user_id")["event_time"]
        .min()
        .reset_index(name="install_time")
    )
    # Merge install time
    df = df.merge(installs, on="user_id", how="left")
    df["days_after_install"] = (df["event_time"] - df["install_time"]).dt.days

    churn_users = (
        df[(df["days_after_install"] >= 30) & (df["days_after_install"] <= 90)]
        .groupby("user_id")
        .size()
        .reset_index(name="has_event")
    )
    churn_users["churn"] = 0  # retained

    labels = installs[["user_id"]].copy()
    labels = labels.merge(
        churn_users[["user_id", "churn"]], how="left", on="user_id"
    )
    labels["churn"] = labels["churn"].fillna(1).astype(int)

    return labels, installs

# --- Feature Engineering Function ---
def featurize(df, installs, labels=None):
    # Merge install_time per user
    df = df.merge(installs, on="user_id", how="left")
    df["days_after_install"] = (df["event_time"] - df["install_time"]).dt.days

    # Keep only events in first 0–7 days after install
    df7 = df[df["days_after_install"] <= 7].copy()

    # Base aggregations
    f = df7.groupby("user_id").agg(
        total_events_7d=("event_type", "count"),
        distinct_event_types_7d=("event_type", pd.Series.nunique),
        total_sessions_7d=("session_id", pd.Series.nunique),
        active_days_7d=("event_time", lambda x: x.dt.date.nunique()),
        avg_events_per_session_7d=("session_id", lambda x: len(x)/x.nunique() if x.nunique()>0 else 0),
        events_per_active_day=("event_time", lambda x: len(x)/x.dt.date.nunique() if x.dt.date.nunique()>0 else 0),
        paying_user=("paying", lambda x: x.astype(str).str.lower().eq("true").astype(int).mean()),
    ).reset_index()

    # time_to_first_session_min
    firsts = (df7.sort_values("event_time")
                 .groupby(["user_id", "session_id"])["event_time"]
                 .min()
                 .reset_index())
    installs_sessions = firsts[firsts["event_time"] == firsts["event_time"].groupby(firsts["user_id"]).transform("min")]
    next_sessions = firsts.merge(
        installs_sessions[["user_id", "session_id"]],
        on="user_id", how="left", suffixes=("", "_inst")
    ).query("session_id != session_id_inst")
    first_valid = (next_sessions.sort_values("event_time")
                   .groupby("user_id")["event_time"]
                   .first()
                   .reset_index(name="first_valid_session"))
    tfs = installs.merge(first_valid, on="user_id", how="left")
    tfs["time_to_first_session_min"] = (
        (tfs["first_valid_session"] - tfs["install_time"])
        .dt.total_seconds() / 60
    ).fillna(0)
    f = f.merge(tfs[["user_id", "time_to_first_session_min"]], on="user_id", how="left")

    # session gap
    starts = (df7.groupby(["user_id", "session_id"])["event_time"]
              .min().reset_index(name="start"))
    starts = starts.sort_values(["user_id", "start"])
    starts["prev"] = starts.groupby("user_id")["start"].shift(1)
    starts["gap_min"] = ((starts["start"] - starts["prev"]).dt.total_seconds()/60)
    gap = starts.groupby("user_id")["gap_min"].mean().reset_index(name="time_between_sessions_avg_min")
    f = f.merge(gap, on="user_id", how="left").fillna({"time_between_sessions_avg_min": 0})

    # core event count
    core = ["session_start", "whats_new_screen_shown", "forecast_tab_shown", "radar_tab_shown", "splash_screen_started"]
    core_cnt = (df7[df7["event_type"].isin(core)]
                .groupby("user_id")["event_type"]
                .count()
                .reset_index(name="core_event_interaction_count_7d"))
    f = f.merge(core_cnt, on="user_id", how="left").fillna({"core_event_interaction_count_7d": 0})

    # session durations
    dur = (df7.groupby(["user_id", "session_id"])["event_time"]
           .agg(["min", "max"])
           .reset_index())
    dur["dur_min"] = ((dur["max"] - dur["min"]).dt.total_seconds()/60)
    stats = (dur.groupby("user_id")["dur_min"]
             .agg(session_duration_mean_min="mean",
                  session_duration_max_min="max",
                  session_duration_total_min="sum")
             .reset_index())
    f = f.merge(stats, on="user_id", how="left").fillna({
        "session_duration_mean_min": 0,
        "session_duration_max_min": 0,
        "session_duration_total_min": 0
    })

    # notifications_enabled
    notif = (df7.groupby("user_id")["event_type"]
             .apply(lambda x: int("notification_permission_allowed" in x.values))
             .reset_index(name="notifications_enabled"))
    f = f.merge(notif, on="user_id", how="left").fillna({"notifications_enabled": 0})

    # early_monetization
    em = (df7.groupby("user_id")["paying"]
        .apply(lambda x: int(x.astype(str).str.lower().eq("true").any()))
        .reset_index(name="early_monetization"))
    f = f.merge(em, on="user_id", how="left").fillna({"early_monetization": 0})

    # events_per_active_day
    f["events_per_active_day"] = f["total_events_7d"] / f["active_days_7d"].replace(0, 1)

    # distinct_events_per_session
    des = (df7.groupby(["user_id", "session_id"])["event_type"]
           .nunique().reset_index())\
        .groupby("user_id")["event_type"].mean()\
        .reset_index(name="distinct_events_per_session")
    f = f.merge(des, on="user_id", how="left").fillna({"distinct_events_per_session": 0})

    # days_with_notifications
    days = (df7[df7["event_type"].isin(
        ["notification_received", "notification_opened", "notification_permission_allowed"]
    )]
        .groupby("user_id")["event_time"]
        .apply(lambda x: x.dt.date.nunique())
        .reset_index(name="days_with_notifications"))
    f = f.merge(days, on="user_id", how="left").fillna({"days_with_notifications": 0})

    # max_gap_days
    ss = df7.drop_duplicates(["user_id", "session_id"])\
        .sort_values(["user_id", "event_time"])
    ss["date"] = ss["event_time"].dt.date
    mg = (ss.groupby("user_id")["date"]
          .apply(lambda d: max([(b-a).days for a, b in zip(sorted(d)[:-1], sorted(d)[1:])] or [0]))
          .reset_index(name="max_gap_days"))
    f = f.merge(mg, on="user_id", how="left").fillna({"max_gap_days": 0})

    # event_density, core_event_ratio, monetization_latency
    f["event_density"] = f["total_events_7d"] / f["active_days_7d"].replace(0, 1)
    f["core_event_ratio"] = f["core_event_interaction_count_7d"] / f["total_events_7d"].replace(0, 1)
    f["monetization_latency"] = f.apply(
        lambda r: r["time_to_first_session_min"] if r["early_monetization"] == 1 else -1,
        axis=1
    )

    # merge churn label if available
    if labels is not None:
        f = f.merge(labels, on="user_id", how="inner")

    return f

input_path = "/opt/ml/processing/input"
if os.path.exists(input_path):
    print("DEBUG: ls -l", input_path, "=", os.listdir(input_path))
    for d in os.listdir(input_path):
        full_path = os.path.join(input_path, d)
        if os.path.isdir(full_path):
            print("DEBUG: ", d, ":", os.listdir(full_path))
        else:
            print(f"DEBUG: skipping {d} because it is not a directory")
else:
    print(f"DEBUG: {input_path} does not exist (likely running locally)")


def resolve_latest_parquet_input(input_dir):
   
    fs = s3fs.S3FileSystem(anon=False)

    if input_dir.startswith("s3://"):
        if input_dir.endswith(".parquet"):
            return input_dir

        all_files = fs.ls(input_dir, detail=True)

        date_folders = []
        pattern = re.compile(r".*/(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$")
        for f in all_files:
            key = f["Key"] if "Key" in f else f["name"]
            folder_candidate = key.rstrip('/').split('/')[-1]
            if pattern.match(key) and re.match(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", folder_candidate):
                date_folders.append(key)

        if not date_folders:
            raise ValueError(f"No timestamped folders found in {input_dir}")

        latest_folder = sorted(date_folders)[-1]
        latest_files = fs.ls(f"s3://{latest_folder}", detail=True)
        parquet_files = [
            f"s3://{file['Key']}" for file in latest_files
            if file["Key"].endswith(".parquet")
        ]
        if not parquet_files:
            raise ValueError(f"No parquet files found in {latest_folder}")
        return sorted(parquet_files)[-1]

    else:
        import glob, os
        if os.path.isfile(input_dir) and input_dir.endswith(".parquet"):
            return input_dir
        elif os.path.isdir(input_dir):
            subfolders = [os.path.join(input_dir, d)
                          for d in os.listdir(input_dir)
                          if os.path.isdir(os.path.join(input_dir, d)) and d.startswith("20")]
            if not subfolders:
                raise ValueError(f"No data subfolders found in {input_dir}")
            latest_folder = sorted(subfolders)[-1]
            parquet_files = glob.glob(os.path.join(latest_folder, "*.parquet"))
            if not parquet_files:
                raise ValueError(f"No parquet files found in {latest_folder}")
            return parquet_files[0]
        else:
            raise ValueError(f"Invalid input_dir: {input_dir}")
        

# --- Read a single parquet file (local or S3) ---
def read_parquet_file(input_path):
    """Read a single parquet file from local or S3."""
    if input_path.startswith("s3://"):
        return pd.read_parquet(input_path, storage_options={'anon': False})
    else:
        return pd.read_parquet(input_path)

# --- Main ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True, help="S3 or local folder or parquet file; if folder, will auto-select latest.")
    parser.add_argument("--output-dir", type=str, required=True, help="S3 or local path to write features.parquet")
    parser.add_argument("--install-file", type=str, required=True, help="S3 or local path to install times CSV")
    parser.add_argument("--mode", choices=["train", "inference"], default="inference", help="train: include churn label; inference: no label")
    args = parser.parse_args()

    # --- Find and read latest parquet file ---
    resolved_input_file = resolve_latest_parquet_input(args.input_dir)
    print(f"Using input parquet file: {resolved_input_file}")
    df = read_parquet_file(resolved_input_file)
    print(f"Loaded input shape: {df.shape}")

    # --- Clean events ---
    df = clean_events(df)
    print(f"After cleaning: {df.shape}")

    install_df = pd.read_csv(args.install_file)
    install_df["estimated_install_time"] = pd.to_datetime(install_df["estimated_install_time"])

    # --- Label churn or just calculate installs ---
    if args.mode == "train":
        labels, installs = label_churn(df, install_df)
    else:
        installs = df.groupby("user_id")["event_time"].min().reset_index(name="install_time")
        labels = None

    # --- Feature engineering ---
    features = featurize(df, installs, labels)
    print(f"Final features shape: {features.shape}")

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = f"{args.output_dir.rstrip('/')}/{run_id}/features.parquet"

    if out_path.startswith("s3://"):
        features.to_parquet(out_path, index=False, storage_options={'anon': False})
    else:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        features.to_parquet(out_path, index=False)

    print("Wrote features →", out_path)

