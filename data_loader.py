import fastf1
import os
import pandas as pd
import time

# Enabling FastF1 cache
cache_dir = "cache_folder"
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

# Creating folder for .csv files
csv_dir = "csv_data"
os.makedirs(csv_dir, exist_ok=True)

# Fetch and format session results
def _fetch_session_data(year, event_name, session_type):

    session = fastf1.get_session(year, event_name, session_type)
    session.load(telemetry=False, weather=False, messages=False)
    df = session.results.copy()
    df["Year"] = year
    df["EventName"] = event_name
    return df

# Downloading set year
def download_season(year):

    race_file = f"{csv_dir}/race_{year}.csv"
    quali_file = f"{csv_dir}/quali_{year}.csv"
    sprint_file = f"{csv_dir}/sprint_{year}.csv"
    
    # Checking if data is already downloaded
    has_base = os.path.exists(race_file) and os.path.exists(quali_file)
    has_sprint = os.path.exists(sprint_file)
    cache_complete = has_base and (year < 2024 or has_sprint)
            
    if cache_complete:
        print(f"Data for season {year} loaded from cache.")
        def safe_read(f):
            return pd.read_csv(f) if os.path.exists(f) and os.path.getsize(f) > 0 else pd.DataFrame()
        return safe_read(race_file), safe_read(quali_file), safe_read(sprint_file)
        
    print(f"Downloading data for season {year} from FastF1 API")
    schedule = fastf1.get_event_schedule(year)
    results = {"R": [], "Q": [], "S": []}
    season_success = True
    
    for event in schedule.itertuples():

        # Skipping pre-season testing and races that haven't taken place yet
        if event.EventFormat == "testing" or event.EventDate > pd.Timestamp.now(tz=event.EventDate.tz):
            continue
            
        try:
            results["R"].append(_fetch_session_data(year, event.EventName, "R"))
            results["Q"].append(_fetch_session_data(year, event.EventName, "Q"))
            
            # Fetching sprint data from 2024 onwards using EventFormat
            if year >= 2024 and "sprint" in str(getattr(event, "EventFormat", "")).lower():
                try:
                    results["S"].append(_fetch_session_data(year, event.EventName, "S"))
                except Exception as e_s:
                    if "calls/h" in str(e_s) or "limit" in str(e_s).lower():
                        raise e_s  # Pass rate limit error to the outer block
            
        # Exeption for hitting API limit
        except Exception as e:
            print(f"Error downloading {event.EventName} {year}: {e}")
            if "calls/h" in str(e) or "limit" in str(e).lower():
                print("API limit reached. Stopping download.")
                season_success = False
                break
                
        time.sleep(0.5)
            
    df_race = pd.concat(results["R"], ignore_index=True) if results["R"] else pd.DataFrame()
    df_quali = pd.concat(results["Q"], ignore_index=True) if results["Q"] else pd.DataFrame()
    df_sprint = pd.concat(results["S"], ignore_index=True) if results["S"] else pd.DataFrame()
    
    # Saving to CSV
    if season_success and not df_race.empty and not df_quali.empty:
        df_race.to_csv(race_file, index=False)
        df_quali.to_csv(quali_file, index=False)
        if not df_sprint.empty:
            df_sprint.to_csv(sprint_file, index=False)
        print(f"Data for {year} successfully saved to CSV.")
    elif not season_success:
        print(f"CSV not saved for {year} due to API limit.")
        
    return df_race, df_quali, df_sprint

def _concat_dfs(df_list):

    return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

def load_all_data():
    
    # Automatically determine the current year and the range for historical data
    current_year = pd.Timestamp.now().year
    historical_years = list(range(2018, current_year))
    
    hist_results = {"R": [], "Q": [], "S": []}
    
    print(f"Fetching historical data for years: {historical_years}")
    for year in historical_years:
        r, q, s = download_season(year)
        hist_results["R"].append(r)
        hist_results["Q"].append(q)
        hist_results["S"].append(s)
        
    # Creating 1 DataFrame for each session type
    df_historical_races = _concat_dfs(hist_results["R"])
    df_historical_qualis = _concat_dfs(hist_results["Q"])
    df_historical_sprints = _concat_dfs(hist_results["S"])
    
    print(f"\nFetching current season data for year: {current_year}")
    df_current_races, df_current_qualis, df_current_sprints = download_season(current_year)
    
    print(f"\nHistorical seasons | Races: {len(df_historical_races)} | Qualis: {len(df_historical_qualis)} | Sprints: {len(df_historical_sprints)}")
    print(f"Current season | Races: {len(df_current_races)} | Qualis: {len(df_current_qualis)} | Sprints: {len(df_current_sprints)}")
    
    return (df_historical_races, df_historical_qualis, df_historical_sprints,
            df_current_races, df_current_qualis, df_current_sprints)