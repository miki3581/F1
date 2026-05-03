import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Calculating team form 
def calculate_team_rolling_points(df, window=3):

    df_fe = df.copy()

    if "SprintPoints" not in df_fe.columns:
        df_fe["SprintPoints"] = 0.0
        
    # Calculating total event points
    df_fe["TotalEventPoints"] = df_fe["Points"].fillna(0) + df_fe["SprintPoints"].fillna(0)

    # Aggregate total points scored by each team in each race
    team_points = df_fe.groupby(['TeamName', 'Year', 'EventName'], sort=False)['TotalEventPoints'].sum().reset_index()

    # Calculating the rolling average of total team points
    team_points['TeamRollingAvgPoints'] = (
        team_points.groupby('TeamName')['TotalEventPoints']
        .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    )

    # Filling first race with 0.0 points
    team_points['TeamRollingAvgPoints'] = team_points['TeamRollingAvgPoints'].fillna(0.0)

    # Merging with main dataframe
    team_points.drop(columns=['TotalEventPoints'], inplace=True) 
    df_fe = pd.merge(df_fe, team_points, on=['TeamName', 'Year', 'EventName'], how='left')
    df_fe.drop(columns=['TotalEventPoints'], inplace=True, errors='ignore')
    
    return df_fe

# Calculating driver form
def calculate_driver_rolling_points(df, window=3):
    
    df_fe = df.copy()

    if "SprintPoints" not in df_fe.columns:
        df_fe["SprintPoints"] = 0.0
        
    # Calculating total event points
    df_fe["TotalEventPoints"] = df_fe["Points"].fillna(0) + df_fe["SprintPoints"].fillna(0)
    
    # Calculating rolling average of total driver points
    df_fe["DriverRollingAvgPoints"] = (
        df_fe.groupby("FullName")["TotalEventPoints"]
        .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    )
    
    # If a driver has no history, assume they perform at their current Team's average (Team Points / 2)
    df_fe["DriverRollingAvgPoints"] = df_fe["DriverRollingAvgPoints"].fillna(df_fe["TeamRollingAvgPoints"] / 2)
    
    df_fe.drop(columns=["TotalEventPoints"], inplace=True, errors='ignore')
    
    return df_fe

def calculate_driver_rolling_position(df, window=3):
    
    df_fe = df.copy()
    
    # Creating a temporary position column that sets Mechanical DNFs to NaN
    df_fe["_ValidPosition"] = df_fe["Position"]
    df_fe.loc[df_fe["StatusCategory"] == "Mechanical", "_ValidPosition"] = np.nan
    
    # Grouping by driver and calculating the rolling average valid positions
    df_fe["DriverRollingAvgPosition"] = (
        df_fe.groupby("FullName")["_ValidPosition"]
        .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    )
    
    # Calculating Team's average finishing position for fallback
    team_pos = df_fe.groupby(['TeamName', 'Year', 'EventName'], sort=False)['_ValidPosition'].mean().reset_index()
    team_pos['TeamRollingAvgPosition'] = (
        team_pos.groupby('TeamName')['_ValidPosition']
        .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    )

    team_pos['TeamRollingAvgPosition'] = team_pos['TeamRollingAvgPosition'].fillna(10.0)
    
    # Merging data
    df_fe = pd.merge(df_fe, team_pos[['TeamName', 'Year', 'EventName', 'TeamRollingAvgPosition']], on=['TeamName', 'Year', 'EventName'], how='left')
    
    # Filling missing rookie positions with the Team's average position
    df_fe["DriverRollingAvgPosition"] = df_fe["DriverRollingAvgPosition"].fillna(df_fe["TeamRollingAvgPosition"])
    df_fe.drop(columns=["_ValidPosition", "TeamRollingAvgPosition"], inplace=True)
    
    return df_fe

def calculate_track_affinity(df, window=3):
    
    df_fe = df.copy()
    
    if "SprintPoints" not in df_fe.columns:
        df_fe["SprintPoints"] = 0.0
        
    # Calculating total event points
    df_fe["TotalEventPoints"] = df_fe["Points"].fillna(0) + df_fe["SprintPoints"].fillna(0)
    
    # Calculatign driver's historical average points at this specific track
    df_fe["DriverTrackAvgPoints"] = (
        df_fe.groupby(["FullName", "EventName"])["TotalEventPoints"]
        .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    )
    
    # If new to track track average is equal to driver average
    df_fe["DriverTrackAvgPoints"] = df_fe["DriverTrackAvgPoints"].fillna(df_fe["DriverRollingAvgPoints"])
    
    # Driver's historical average position at this specific track (ignoring mechanical DNFs)
    df_fe["_ValidPosition"] = df_fe["Position"]
    df_fe.loc[df_fe["StatusCategory"] == "Mechanical", "_ValidPosition"] = np.nan
    
    # Calculating Driver's average finishing position for fallback
    df_fe["DriverTrackAvgPosition"] = (
        df_fe.groupby(["FullName", "EventName"])["_ValidPosition"]
        .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    )

    # If new to the track, fallback to their general current form
    df_fe["DriverTrackAvgPosition"] = df_fe["DriverTrackAvgPosition"].fillna(df_fe["DriverRollingAvgPosition"])
    
    df_fe.drop(columns=["TotalEventPoints"], inplace=True, errors='ignore')
    df_fe.drop(columns=["_ValidPosition"], inplace=True)
    
    return df_fe

def get_current_form(df, window=3):

    df_fe = df.copy()
    if "SprintPoints" not in df_fe.columns:
        df_fe["SprintPoints"] = 0.0
        
    # Calculating total event points
    df_fe["TotalEventPoints"] = df_fe["Points"].fillna(0) + df_fe["SprintPoints"].fillna(0)

    # Rolling averages going into the upcoming race
    # Driver points
    driver_pts = df_fe.groupby("FullName")["TotalEventPoints"].apply(lambda x: x.tail(window).mean()).reset_index(name="Next_DriverAvgPoints")
    
    # Driver Position
    df_valid = df.copy()
    df_valid.loc[df_valid["StatusCategory"] == "Mechanical", "Position"] = np.nan
    driver_pos = df_valid.groupby("FullName")["Position"].apply(lambda x: x.dropna().tail(window).mean()).reset_index(name="Next_DriverAvgPos")
    
    # Team Points
    team_race_pts = df_fe.groupby(['TeamName', 'Year', 'EventName'], sort=False)['TotalEventPoints'].sum().reset_index()
    team_pts = team_race_pts.groupby("TeamName")["TotalEventPoints"].apply(lambda x: x.tail(window).mean()).reset_index(name="Next_TeamAvgPoints")
    
    # Map drivers to their latest team
    latest_teams = df.drop_duplicates(subset=["FullName"], keep="last")[["FullName", "TeamName"]]
    
    # Combine everything and round to 2 decimal places for a clean printout
    current_form = pd.merge(driver_pts, driver_pos, on="FullName")
    current_form = pd.merge(current_form, latest_teams, on="FullName")
    current_form = pd.merge(current_form, team_pts, on="TeamName").round(2)
    
    return current_form.sort_values(by="Next_DriverAvgPoints", ascending=False)

def encode_categorical_features(df):
    
    df_encoded = df.copy()

    # Chosing cols to encode
    categorical_cols = ["EventName", "FullName", "TeamName"]
    
    # One-Hot encoding
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    # Fit and transform the data, converting numpy array back to a dataframe
    encoded_array = encoder.fit_transform(df_encoded[categorical_cols])
    encoded_cols = encoder.get_feature_names_out(categorical_cols)
    encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=df_encoded.index)
    
    # Droping original categorical columns and concating new one-hot encoded columns
    df_encoded = df_encoded.drop(columns=categorical_cols)
    df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
    
    # Droping intermediate columns
    df_encoded.drop(columns=["Status", "StatusCategory"], errors="ignore", inplace=True)
    
    return df_encoded
