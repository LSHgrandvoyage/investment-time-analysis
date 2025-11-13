import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')

from matplotlib.dates import DateFormatter
from datetime import datetime
from dotenv import load_dotenv

def load_data():
    '''
    Load JSON signal data && Preprocess datetime fields

    return: pd.DataFrame -> Preprocessed dataframe (datetime, date, weekday, time_slot)
    '''
    load_dotenv()
    data_dir = os.getenv('DATA_PATH')
    with open(data_dir, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = pd.to_datetime(df['date']).dt.date
    df['weekday'] = df['datetime'].dt.day_name()
    df['time_slot'] = pd.to_datetime(df['time_slot'])
    df['time_slot'] = df['time_slot'].dt.floor('30min')
    return df

def signal_timeline(df):
    '''
    Plot a scatter timeline of signal occurrences per day
    param : df (pd.DataFrame)
    '''
    plt.figure(figsize=(12, 8))
    unique_dates = sorted(df['date'].unique())

    for i, date in enumerate(unique_dates):
        times = df[df['date'] == date]['datetime'].dt.hour + df[df['date'] == date]['datetime'].dt.minute / 60
        plt.scatter(times, [i] * len(times), s=50, alpha=0.6, label=str(date))

        signal_count = len(times)
        plt.text(24.5, i, f'{signal_count}', va='center', fontsize=10, color='red')

    plt.yticks(range(len(unique_dates)), unique_dates)
    plt.xlabel("Hour of day")
    plt.ylabel("Date")
    plt.title("Timeline of Signals per Day (Number of signals on right)")
    plt.xlim(0, 25)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("signal_timeline_with_count.png")
    plt.close()

# Hourly histogram of signals
def signal_histogram(df):
    '''
    Plot a histogram showing total siganl counts per hour
    param: df (pd.DataFrame)
    '''
    plt.figure(figsize=(12,6))
    hour_counts = df['datetime'].dt.hour.value_counts().sort_index()
    bars = plt.bar(hour_counts.index, hour_counts.values, color='skyblue', alpha=0.7)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.2, str(int(height)),
                 ha='center', va='bottom', fontsize=10, color='red')

    plt.xlabel("Hour of Day")
    plt.ylabel("Number of Signals")
    plt.title("Number of Signals per Hour")
    plt.xticks(range(0, 24))
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("signal_hour_histogram_with_count.png")
    plt.close()

# Weekday average signals
def avg_signal_weekday(df):
    '''
    Plot the average number of signals per weekday with overall daily average line
    param: df (pd.DataFrame)
    '''
    weekday_count = df.groupby('weekday').size()
    weekday_days = df.groupby('weekday')['date'].nunique()
    weekday_avg = (weekday_count / weekday_days).sort_values(ascending=True)
    daily_avg = df.groupby('date').size().mean()

    plt.figure(figsize=(8, 4))
    bars = plt.bar(weekday_avg.index, weekday_avg.values, color='lightgreen', alpha=0.7)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.05, f'{height:.1f}',
                 ha='center', va='bottom', fontsize=10, color='red')

    plt.axhline(daily_avg, color='blue', linestyle='-', linewidth=2, label=f'Daily Avg: {daily_avg:.1f}')
    plt.text(len(weekday_avg) - 0.5, daily_avg + 0.05, f'Daily Avg: {daily_avg:.1f}',
             color='blue', fontsize=10, ha='right', va='bottom')

    plt.title('Average Number of Signals per Weekday')
    plt.xlabel('Weekday')
    plt.ylabel('Average Signals')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("weekday_avg_signals_with_count.png")
    plt.close()

def find_profitable_time(df, window, bin_size=0.5, avoid_time=None):
    '''
    Find the time window with the fewest average signal occurrences across all days.
    Optionally excludes overlap with an avoid_time period

    param: df (pd.DataFrame)
    param: window (Duration of window in hours)
    param: bin_size (Size of time bins in hours)
    param: avoid_time (Tuple (start_hour, end_hour) to exclude from search)

    return: tuple (start_hour, end_hour, expected_loss_signal)
    '''
    df = df.copy()
    df['hour_decimal'] = df['datetime'].dt.hour + df['datetime'].dt.minute / 60

    time_bins = np.arange(0, 24 + bin_size, bin_size)
    window_bins = int(window / bin_size)
    n_bins = len(time_bins) - 1

    unique_dates = df['date'].unique()
    avg_signals_per_window = []

    for start_bin in range(n_bins):
        start_hour = start_bin * bin_size
        end_hour = (start_hour + window) % 24

        if avoid_time is not None:
            avoid_start, avoid_end = avoid_time
            overlap = False
            if avoid_start < avoid_end:
                overlap = not (end_hour <= avoid_start or start_hour >= avoid_end)
            else:
                overlap = not (end_hour <= avoid_start and start_hour >= avoid_end)
            if overlap:
                avg_signals_per_window.append(np.inf)
                continue

        total_signals = 0
        for date in unique_dates:
            day_hours = df.loc[df['date'] == date, 'hour_decimal'].values
            counts, _ = np.histogram(day_hours, bins=time_bins)
            counts = np.concatenate([counts, counts])
            window_sum = counts[start_bin:start_bin + window_bins].sum()
            total_signals += window_sum

        avg_signal = total_signals / len(unique_dates)
        avg_signals_per_window.append(avg_signal)

    min_idx = np.argmin(avg_signals_per_window)
    start_hour = min_idx * bin_size
    end_hour = (start_hour + window) % 24
    expected_loss_signal = avg_signals_per_window[min_idx]

    return start_hour, end_hour, expected_loss_signal

def hour_to_str(hour):
    '''
    Convert a float hour to HH:MM string

    param: hour (Hour in decimal format)

    return: str (Time string in HH:MM format)
    '''
    h = int(hour)
    m = int((hour - h) * 60)
    return f"{h:02d}:{m:02d}"

if __name__ == '__main__':
    df = load_data()
    signal_timeline(df)
    signal_histogram(df)
    avg_signal_weekday(df)

    start, end, signal = find_profitable_time(df, 7)
    print("==============================================")
    print("Sleep start : " + hour_to_str(start))
    print("Sleep end : " + hour_to_str(end))
    print("Expected loss signals : " + str(signal)[:4])
    print("==============================================")

    start, end, signal = find_profitable_time(df, 1.5, avoid_time=(start, end))
    print("Break start : " + hour_to_str(start))
    print("Break end : " + hour_to_str(end))
    print("Expected loss signals : " + str(signal)[:4])
    print("==============================================")
