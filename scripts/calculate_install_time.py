# build_full_install_time.py

import pandas as pd
import pyarrow.dataset as ds

EVENT_PATH = "s3://analytics-v0-501209598921-eu-central-1-amplitude/bronze/user_events/application_id=NORD/"

print("Loading all event data from S3...")
dataset = ds.dataset(EVENT_PATH, format="parquet")

tbl = dataset.to_table(columns=["user_id", "event_time"])
print(f"Loaded events: {tbl.num_rows:,} rows.")

# Convert to DataFrame
df = tbl.to_pandas()
df["event_time"] = pd.to_datetime(df["event_time"])

install_df = (
    df.groupby("user_id")["event_time"]
      .min()
      .reset_index()
      .rename(columns={"event_time": "estimated_install_time"})
)

print(f"Unique users found: {install_df.shape[0]:,}")

install_df.to_csv("NORD-install_time_full.csv", index=False)
print("Wrote full install time file to: New-NORD-install_time_full.csv")