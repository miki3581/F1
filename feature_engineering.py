import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def calculate_driver_rolling_points(df, window=3):
    
    df_fe = df.copy()
    
    # We group by driver and calculate the rolling mean of the 'Points' column.
    # shift(1) to look only at past races and don't leak the current race's points
    df_fe["DriverRollingAvgPoints"] = (
        df_fe.groupby("FullName")["Points"]
        .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    )
    
    # For a driver's very first race in the dataset, the rolling average will be NaN.
    # We fill this with 0.0 to indicate no prior points.
    df_fe["DriverRollingAvgPoints"] = df_fe["DriverRollingAvgPoints"].fillna(0.0)
    
    return df_fe

def calculate_driver_rolling_position(df, window=3):
    
    df_fe = df.copy()
    
    # Create a temporary position column that sets Mechanical DNFs to NaN
    df_fe["_ValidPosition"] = df_fe["Position"]
    df_fe.loc[df_fe["StatusCategory"] == "Mechanical", "_ValidPosition"] = np.nan
    
    # Group by driver and calculate the rolling mean on the valid positions
    df_fe["DriverRollingAvgPosition"] = (
        df_fe.groupby("FullName")["_ValidPosition"]
        .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    )
    
    # Fill the first race with an average mid-pack position (10.0)
    df_fe["DriverRollingAvgPosition"] = df_fe["DriverRollingAvgPosition"].fillna(10.0)
    df_fe.drop(columns=["_ValidPosition"], inplace=True)
    
    return df_fe

def calculate_team_rolling_points(df, window=3):
    df_fe = df.copy()

    # Aggregate total points scored by each team in each race
    team_points = df_fe.groupby(['TeamName', 'Year', 'EventName'], sort=False)['Points'].sum().reset_index()

    # Calculate the rolling average of these total team points
    team_points['TeamRollingAvgPoints'] = (
        team_points.groupby('TeamName')['Points']
        .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    )

    # Fill the first race with 0.0 points
    team_points['TeamRollingAvgPoints'] = team_points['TeamRollingAvgPoints'].fillna(0.0)
    
    # Merge this new feature back into our main driver-level dataframe
    team_points.drop(columns=['Points'], inplace=True) # Drop so it doesn't conflict with driver Points
    df_fe = pd.merge(df_fe, team_points, on=['TeamName', 'Year', 'EventName'], how='left')
    
    return df_fe

def calculate_track_affinity(df, window=3):
    
    df_fe = df.copy()
    
    # Driver's historical average points at this specific track
    df_fe["DriverTrackAvgPoints"] = (
        df_fe.groupby(["FullName", "EventName"])["Points"]
        .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    )
    df_fe["DriverTrackAvgPoints"] = df_fe["DriverTrackAvgPoints"].fillna(0.0)
    
    # Driver's historical average position at this specific track (ignoring mechanical DNFs)
    df_fe["_ValidPosition"] = df_fe["Position"]
    df_fe.loc[df_fe["StatusCategory"] == "Mechanical", "_ValidPosition"] = np.nan
    
    df_fe["DriverTrackAvgPosition"] = (
        df_fe.groupby(["FullName", "EventName"])["_ValidPosition"]
        .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    )
    df_fe["DriverTrackAvgPosition"] = df_fe["DriverTrackAvgPosition"].fillna(10.0)
    df_fe.drop(columns=["_ValidPosition"], inplace=True)
    
    return df_fe

def get_current_form(df, window=3):

    # Rolling averages going into the upcoming race
    # Driver points
    driver_pts = df.groupby("FullName")["Points"].apply(lambda x: x.tail(window).mean()).reset_index(name="Next_DriverAvgPoints")
    
    # Driver Position
    df_valid = df.copy()
    df_valid.loc[df_valid["StatusCategory"] == "Mechanical", "Position"] = np.nan
    driver_pos = df_valid.groupby("FullName")["Position"].apply(lambda x: x.dropna().tail(window).mean()).reset_index(name="Next_DriverAvgPos")
    
    # Team Points
    team_race_pts = df.groupby(['TeamName', 'Year', 'EventName'], sort=False)['Points'].sum().reset_index()
    team_pts = team_race_pts.groupby("TeamName")["Points"].apply(lambda x: x.tail(window).mean()).reset_index(name="Next_TeamAvgPoints")
    
    # Map drivers to their latest team
    latest_teams = df.drop_duplicates(subset=["FullName"], keep="last")[["FullName", "TeamName"]]
    
    # Combine everything and round to 2 decimal places for a clean printout
    current_form = pd.merge(driver_pts, driver_pos, on="FullName")
    current_form = pd.merge(current_form, latest_teams, on="FullName")
    current_form = pd.merge(current_form, team_pts, on="TeamName").round(2)
    
    return current_form.sort_values(by="Next_DriverAvgPoints", ascending=False)

def encode_categorical_features(df):
    
    df_encoded = df.copy()
    categorical_cols = ["EventName", "FullName", "TeamName"]
    
    # Initialize the Scikit-Learn OneHotEncoder
    # handle_unknown='ignore' ensures that if a new driver/team appears in the future, it won't crash
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    # Fit and transform the data, converting the resulting numpy array back to a DataFrame
    encoded_array = encoder.fit_transform(df_encoded[categorical_cols])
    encoded_cols = encoder.get_feature_names_out(categorical_cols)
    encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=df_encoded.index)
    
    # Drop original categorical columns and concatenate the new one-hot encoded columns
    df_encoded = df_encoded.drop(columns=categorical_cols)
    df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
    
    # Drop intermediate columns that would cause Data/Target Leakage
    df_encoded.drop(columns=["Status", "StatusCategory"], errors="ignore", inplace=True)
    
    return df_encoded
