import data_loader
import preprocessing
import feature_engineering
import model
import pandas as pd

def main():

    # Loading data
    (df_hist_races, df_hist_qualis, df_hist_sprints,
     df_curr_races, df_curr_qualis, df_curr_sprints) = data_loader.load_all_data()

    # Cleaning data
    df_hist_races_clean = preprocessing.clean_race_data(df_hist_races)
    df_hist_qualis_clean = preprocessing.clean_quali_data(df_hist_qualis)
    df_hist_sprints_clean = preprocessing.clean_sprint_data(df_hist_sprints)
    df_curr_races_clean = preprocessing.clean_race_data(df_curr_races)
    df_curr_qualis_clean = preprocessing.clean_quali_data(df_curr_qualis)
    df_curr_sprints_clean = preprocessing.clean_sprint_data(df_curr_sprints)

    # Merging data
    df_hist_merged = preprocessing.merge_data(df_hist_races_clean, df_hist_qualis_clean)
    df_curr_merged = preprocessing.merge_data(df_curr_races_clean, df_curr_qualis_clean)
    
    # Combining historical and current data for Feature Engineering
    df_all = pd.concat([df_hist_merged, df_curr_merged], ignore_index=True)

    # Feature Engineering Pipeline
    df_tp = feature_engineering.calculate_team_rolling_points(df_all, window=3)
    df_dp = feature_engineering.calculate_driver_rolling_points(df_tp, window=3)
    df_dps = feature_engineering.calculate_driver_rolling_position(df_dp, window=3)
    df_ta = feature_engineering.calculate_track_affinity(df_dps, window=3)
    
    # Define the Target Variable for Classification
    df_tar = model.create_target_top5(df_ta)
    
    # Save a human-readable copy before dropping names for One-Hot Encoding
    df_unencoded = df_tar.copy()
    
    # Enocode caterogical features
    df_enc = feature_engineering.encode_categorical_features(df_tar)
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = model.train_test_split_(df_enc)

    # Train and evaluate XGBoost
    xgb_model = model.model_xgb(X_train, X_test, y_train, y_test, tune=True)
    
    # Predict the exact finishing order for the most recent race in our test set
    latest_year = df_unencoded["Year"].max()
    latest_race = df_unencoded[df_unencoded["Year"] == latest_year]["EventName"].unique()[-1]
    model.predict_race_ranking(xgb_model, X_test, df_unencoded, latest_year, latest_race)


if __name__ == "__main__":
    main()
