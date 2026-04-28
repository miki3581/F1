import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt


def create_target_top5(df):
    
    df_target = df.copy()
    # Create binary target: 1 if Position is 1, 2, 3, 4, or 5. Otherwise 0.
    df_target["Target_Top5"] = (df_target["Position"] <= 5).astype(int)
    return df_target

def train_test_split_(df_all):

    df_split = df_all[df_all["Year"] >= 2019].copy()

    y = df_split["Target_Top5"]
    X = df_split.drop(columns=["Target_Top5", "Position", "Points", "Year"])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    return X_train, X_test, y_train, y_test

def model_xgb(X_train, X_test, y_train, y_test, tune=False):

    if tune:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        }
        xgb = XGBClassifier(random_state=42)
        # Optimize for the f1 score to balance precision and recall
        grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring='f1', cv=3, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        
        print(f"\nBest Hyperparameters: {grid_search.best_params_}")
        model = grid_search.best_estimator_
    else:
        model = XGBClassifier(random_state=42)
        model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Top 5", "Top 5"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()
    
    return model

def predict_race_ranking(model, X_test, df_unencoded, year, event_name):
    
    # Find the indices for the specific race
    race_mask = (df_unencoded["Year"] == year) & (df_unencoded["EventName"] == event_name)
    race_indices = df_unencoded[race_mask].index
    
    # Filter X_test to only include this race
    valid_indices = race_indices.intersection(X_test.index)
    
    if len(valid_indices) == 0:
        print(f"\nNo test data found for {event_name} {year}.")
        return
        
    X_race = X_test.loc[valid_indices]
    df_race = df_unencoded.loc[valid_indices]
    
    # Predict probabilities of finishing in Top 5 (Class 1 is at index 1)
    probs = model.predict_proba(X_race)[:, 1]
    
    # Create a results DataFrame
    results = pd.DataFrame({
        "Driver": df_race["FullName"],
        "Team": df_race["TeamName"],
        "Top5_Probability": probs
    })
    
    # Sort by probability descending to get the predicted order
    results = results.sort_values(by="Top5_Probability", ascending=False).reset_index(drop=True)
    results.index = results.index + 1 
    
    print(f"\nPredicted Finishing Order: {event_name} {year}")
    
    # Format probability as a percentage for a clean printout
    results["Top5_Probability"] = (results["Top5_Probability"] * 100).round(1).astype(str) + "%"
    
    # Print the predicted Top 5
    print(results.head(5).to_string())
