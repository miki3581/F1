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

    # Checiking if everything loaded correctly
    print(f"\nHistorical data (Races + Qualis): {df_hist_races.shape[0]}")
    print(f"Historical data (Sprints): {df_hist_sprints.shape[0]}")
    print(f"Current season data (Races + Qualis): {df_curr_races.shape[0]}")
    print(f"Current season data (Sprints): {df_curr_sprints.shape[0]}")
    
    # Combining historical and current data for Feature Engineering
    df_all = pd.concat([df_hist_merged, df_curr_merged], ignore_index=True)

    # Feature Engineering Pipeline
    df_all = feature_engineering.calculate_driver_rolling_points(df_all, window=3)
    df_all = feature_engineering.calculate_driver_rolling_position(df_all, window=3)
    df_all = feature_engineering.calculate_team_rolling_points(df_all, window=3)
    df_all = feature_engineering.calculate_track_affinity(df_all, window=3)
    
    # Define the Target Variable for Classification
    df_all = model.create_target_top5(df_all)
    
    # Enocode caterogical features
    df_all = feature_engineering.encode_categorical_features(df_all)
    


if __name__ == "__main__":
    main()
