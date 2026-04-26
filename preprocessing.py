import pandas as pd

# Helper function to categorize the finish status
def categorize_status(status):
    
    status = str(status).lower()
    if "lap" in status or "finished" in status:
        return "Finished"
    elif any(word in status for word in ["accident", "collision", "spun off", "damage", "front wing"]):
        return "Crash"
    elif any(word in status for word in ["engine", "gearbox", "power", "hydraulics", "suspension", 
                                         "brakes", "electrical", "exhaust", "overheating", "water", 
                                         "fuel", "puncture", "mechanical", "throttle", "wheel", "oil",
                                         "turbo", "electronics", "undertray", "transmission", "vibrations",
                                         "battery", "differential", "leak", "tyre", "front wing", "driveshaft",
                                         "cooling", "overheating", "rear wing", "steering", "radiator", "out of fuel"]):
        return "Mechanical"
    else:
        return "Other DNF"

# Removing unnecesary columns from race
def clean_race_data(df):

    df_clean = df.copy()
    cols_to_drop = [
        "DriverNumber", "BroadcastName", "DriverId", "Abbreviation", 
        "TeamColor", "TeamId", "FirstName", "LastName", "HeadshotUrl", 
        "CountryCode", "Q1", "Q2", "Q3", "Time", "Laps",
        "ClassifiedPosition"
    ]
    df_clean.drop(columns=cols_to_drop, errors="ignore", inplace=True)
    
    # Create the new StatusCategory feature
    if "Status" in df_clean.columns:
        df_clean["StatusCategory"] = df_clean["Status"].apply(categorize_status)
        
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
        
    # Drop raw Q1, Q2, Q3 columns to prevent NaNs for knocked-out drivers
    df_clean.drop(columns=available_q_cols, errors="ignore", inplace=True)
        
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

# Handle missing values (NaNs) for DNFs and missed sessions
def handle_missing_values(df):
    
    df_clean = df.copy()
    
    # Fill missing GridPosition (DNS/Withdrew) and handle pit lane starts 
    if "GridPosition" in df_clean.columns:
        df_clean["GridPosition"] = df_clean["GridPosition"].replace(0.0, 20.0).fillna(20.0)
        
    # Fill missing Race Position (DNS/Withdrew)
    if "Position" in df_clean.columns:
        df_clean["Position"] = df_clean["Position"].fillna(20.0)
        
    # Fill missing Qualifying Positions with their GridPosition 
    if "QualiPosition" in df_clean.columns and "GridPosition" in df_clean.columns:
        df_clean["QualiPosition"] = df_clean["QualiPosition"].fillna(df_clean["GridPosition"])
        
    # Fallback for any remaining NaNs (pitlane starts have GridPosition 0)
    if "QualiPosition" in df_clean.columns:
        df_clean["QualiPosition"] = df_clean["QualiPosition"].replace(0.0, 20.0).fillna(20.0)
        
    # Fill missing BestQualiTime with the slowest time of that specific Event/Year
    if "BestQualiTime" in df_clean.columns:
        df_clean["BestQualiTime"] = df_clean.groupby(["Year", "EventName"])["BestQualiTime"].transform(lambda x: x.fillna(x.max()))
        
    return df_clean

# Pipeline function to clean and merge a set of data
def process_data(df_races, df_qualis, df_sprints):
    
    df_races_clean = clean_race_data(df_races)
    df_qualis_clean = clean_quali_data(df_qualis)
    df_sprints_clean = clean_sprint_data(df_sprints)
    
    return df_races_clean, df_qualis_clean, df_sprints_clean

def merge_data(df_races, df_qualis):

    df_merged = merge_race_and_quali(df_races, df_qualis)
    df_merged_clean = handle_missing_values(df_merged)
    return df_merged_clean