import os
import json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

'''
Sort an origianl JSON data by (date & time).
Add datetime, date, weekday, time_slot info.
Save as new JSON file
'''

load_dotenv()
origin_data_dir = os.getenv("ORIGIN_DATA_PATH")
new_data_dir = os.getenv("DATA_PATH")

with open(origin_data_dir, "r", encoding="utf-8") as f:
    data = json.load(f)

sorted_data = sorted(data, key=lambda x: (x["m"], x["d"], x["h"], x["s"]))

df = pd.DataFrame(sorted_data)

df['datetime'] = df.apply(lambda row: datetime(
    year=2025,
    month=row['m'],
    day=row['d'],
    hour=row['h'],
    minute=row['s']
), axis=1)

df['date'] = df['datetime'].dt.date
df['weekday'] = df['datetime'].dt.day_name()
df['time_slot'] = df['datetime'].dt.floor('30min')

df_final = df[['datetime', 'date', 'weekday', 'time_slot']].copy()

df_final['datetime'] = df_final['datetime'].apply(lambda x: x.isoformat())
df_final['time_slot'] = df_final['time_slot'].apply(lambda x: x.isoformat())
df_final['date'] = df_final['date'].apply(lambda x: x.isoformat())

with open(new_data_dir, "w", encoding="utf-8") as f:
    json.dump(df_final.to_dict(orient='records'), f, indent=2, ensure_ascii=False)
