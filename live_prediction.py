import fastf1
import pandas as pd
import data_loader
import preprocessing
import feature_engineering
import model

def main():
    # Loading historical data
    (df_hist_races, df_hist_qualis, df_hist_sprints,
     df_curr_races, df_curr_qualis, df_curr_sprints) = data_loader.load_all_data()

    df_hist_races_clean = preprocessing.clean_race_data(df_hist_races)
    df_hist_qualis_clean = preprocessing.clean_quali_data(df_hist_qualis)
    df_hist_merged = preprocessing.merge_data(df_hist_races_clean, df_hist_qualis_clean)

    df_curr_races_clean = preprocessing.clean_race_data(df_curr_races)
    df_curr_qualis_clean = preprocessing.clean_quali_data(df_curr_qualis)
    df_curr_merged = preprocessing.merge_data(df_curr_races_clean, df_curr_qualis_clean)

    df_all = pd.concat([df_hist_merged, df_curr_merged], ignore_index=True)

    # Identifing upcoming event
    current_year = pd.Timestamp.now().year
    schedule = fastf1.get_event_schedule(current_year)
    
    upcoming_events = []
    for event in schedule.itertuples():
        if event.EventFormat == "testing":
            continue
        event_date = pd.Timestamp(event.EventDate)
        if event_date > pd.Timestamp.now(tz=event_date.tz):
            upcoming_events.append(event)
            
    if not upcoming_events:
        print("No upcoming events found for the current season.")
        return
        
    next_event = upcoming_events[0]
    event_name = next_event.EventName
    print(f"Next Event: {event_name} {current_year}")

    # Live Qualifying data
    try:
        session_q = fastf1.get_session(current_year, event_name, 'Q')
        session_q.load(telemetry=False, weather=False, messages=False)
        df_live_q = session_q.results.copy()
        df_live_q["Year"] = current_year
        df_live_q["EventName"] = event_name
    except Exception as e:
        print(f"\nCould not load Qualifying data for {event_name}.")
        print(f"Error details: {e}")
        return

    if df_live_q.empty:
        print(f"\nQualifying results for {event_name} are empty. Please try again after the session finishes.")
        return

    # Building a dummy Race dataframe
    df_live_r = df_live_q.copy()
    df_live_r["Position"] = float('nan')
    df_live_r["Points"] = float('nan')
    df_live_r["Status"] = "Unknown"
    
    # GridPosition equals QualiPosition for prediction purposes
    if "Position" in df_live_q.columns:
        df_live_r["GridPosition"] = df_live_q["Position"]
    else:
        df_live_r["GridPosition"] = float('nan')

    # Cleaning and merging live data
    df_live_r_clean = preprocessing.clean_race_data(df_live_r)
    df_live_q_clean = preprocessing.clean_quali_data(df_live_q)
    df_live_merged = preprocessing.merge_data(df_live_r_clean, df_live_q_clean)

    # Appending to the main dataframe
    df_all = pd.concat([df_all, df_live_merged], ignore_index=True)

    # Feature Engineering
    df_tp = feature_engineering.calculate_team_rolling_points(df_all, window=3)
    df_dp = feature_engineering.calculate_driver_rolling_points(df_tp, window=3)
    df_dps = feature_engineering.calculate_driver_rolling_position(df_dp, window=3)
    df_ta = feature_engineering.calculate_track_affinity(df_dps, window=3)
    
    df_tar = model.create_target_top5(df_ta)
    df_unencoded = df_tar.copy()
    df_enc = feature_engineering.encode_categorical_features(df_tar)

    # Split data and train model
    live_mask = (df_unencoded["Year"] == current_year) & (df_unencoded["EventName"] == event_name)
    df_historical = df_enc[~live_mask].copy()
    
    X_train, X_test, y_train, y_test = model.train_test_split_(df_historical)
    xgb_model = model.model_xgb(X_train, X_test, y_train, y_test, tune=False)

    # Predicions
    df_live_features = df_enc[live_mask].copy()
    X_live = df_live_features.drop(columns=["Target_Top5", "Position", "Points", "Year"], errors="ignore")
    
    # Ranking
    model.predict_race_ranking(xgb_model, X_live, df_unencoded, current_year, event_name)

if __name__ == "__main__":
    main()