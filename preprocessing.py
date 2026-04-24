import pandas as pd

# Removing unnecesary columns from race
def clean_race_data(df):

    df_clean = df.copy()
    cols_to_drop = [
        "DriverNumber", "BroadcastName", "DriverId", "Abbreviation", 
        "TeamColor", "TeamId", "FirstName", "LastName", "HeadshotUrl", 
        "CountryCode", "Q1", "Q2", "Q3", "Time", "Points", "Laps",
        "ClassifiedPosition"
    ]
    df_clean.drop(columns=cols_to_drop, inplace=True)
    return df_clean

# Removing unnnecesary columns from quali
def clean_quali_data(df):

    df_clean = df.copy()
    cols_to_drop_quali = [
        "DriverNumber", "BroadcastName", "DriverId", "Abbreviation", 
        "TeamColor", "TeamId", "FirstName", "LastName", "HeadshotUrl", 
        "CountryCode", "ClassifiedPosition", "GridPosition", "Time", 
        "Status", "Points", "Laps"
    ]
    df_clean.drop(columns=cols_to_drop_quali, errors="ignore", inplace=True)
    
    # Formatting qualifying times to total seconds
    for col in ["Q1", "Q2", "Q3"]:
        if col in df_clean.columns:
            df_clean[col] = pd.to_timedelta(df_clean[col]).dt.total_seconds()
            
    # Extracting the fastest time for each driver across all sessions
    available_q_cols = [col for col in ["Q1", "Q2", "Q3"] if col in df_clean.columns]
    if available_q_cols:
        df_clean["BestQualiTime"] = df_clean[available_q_cols].min(axis=1)
        
    return df_clean

# Removing unnecessary columns from sprint
def clean_sprint_data(df):

    df_clean = df.copy()
    cols_to_drop = [
        "DriverNumber", "BroadcastName", "DriverId", "Abbreviation", 
        "TeamColor", "TeamId", "FirstName", "LastName", "HeadshotUrl", 
        "CountryCode", "Q1", "Q2", "Q3", "Time", "Points", "Laps",
        "ClassifiedPosition"
    ]
    df_clean.drop(columns=cols_to_drop, errors="ignore", inplace=True)
    return df_clean

# Merge race with quali based on driver, team, year, and event
def merge_race_and_quali(df_race, df_quali):
    
    # Renaming 'Position' in quali to avoid conflict with the target 'Position' in race
    df_q_renamed = df_quali.rename(columns={"Position": "QualiPosition"})
    
    # Left join ensures we keep all race participants, even if they missed qualifying
    df_merged = pd.merge(df_race, df_q_renamed, on=["FullName", "TeamName", "Year", "EventName"], how="left")
    return df_merged

# Pipeline function to clean and merge a set of data
def process_data(df_races, df_qualis, df_sprints):
    
    df_races_clean = clean_race_data(df_races)
    df_qualis_clean = clean_quali_data(df_qualis)
    df_sprints_clean = clean_sprint_data(df_sprints)
    
    return df_races_clean, df_qualis_clean, df_sprints_clean

def merge_data(df_races, df_qualis):

    df_merged = merge_race_and_quali(df_races, df_qualis)
    return df_merged