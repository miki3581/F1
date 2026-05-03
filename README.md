# Formula 1 Race Predictor 🏎️

## <a name="english"></a>🇬🇧 English

### Overview
This is a Machine Learning project designed to predict Formula 1 race results. By leveraging historical race data and real-time weekend data via the `fastf1` API, the system applies advanced feature engineering and XGBoost models to forecast Top 5 finishes.

### Key Features
- **Data Integration:** Automatic data fetching for historical races, qualifying sessions, and sprints from 2018 until now using `fastf1`.
- **Feature Engineering:** Calculates team and driver rolling forms, track affinities, and integrates sprint race points to capture current momentum. Handles "Did Not Finish" (DNF) statuses by differentiating mechanical failures from crashes.
- **Modeling Approach:** **Classification** Predicts the probability of a driver finishing in the Top 5 using `XGBClassifier`.
- **Live Predictions (`live_prediction.py`):** Automatically identifies the upcoming race weekend, downloads live qualifying (and sprint) data as soon as they finish, and predicts the Sunday Race outcome in real-time.

### Project Structure
- `data_loader.py` - Fetches and loads raw data using FastF1 API.
- `preprocessing.py` - Cleans data, categorizes statuses (e.g., Mechanical vs. Crash), and handles missing variables.
- `feature_engineering.py` - Calculates rolling averages, historical track performances, and current form metrics.
- `model.py` - Handles data splitting (time-series aware), model training, hyperparameter tuning (`GridSearchCV`), and evaluation.
- `main.py` - Entry points to train models and test them on historical data.
- `live_prediction.py` - Real-time pipeline for the upcoming race weekend.

### Requirements
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### How to Run
To train the classification model and predict the latest historical race:
```bash
python main.py
```
To run a real-time prediction for the upcoming Sunday race (run this after Qualifying/Sprint is finished):
```bash
python live_prediction.py
```

## <a name="polski"></a>🇵🇱 Polski

### Opis projektu
Projekt wykorzystujący uczenie maszynowe do przewidywania wyników wyścigów Formuły 1. Dzięki wykorzystaniu danych historycznych oraz danych z trwających weekendów wyścigowych (za pomocą API `fastf1`), system przetwarza zaawansowane cechy statystyczne i używa modeli opartych na algorytmie XGBoost do prognozowania szans na pozycje w Top 5.

### Główne funkcje
- **Integracja danych:** Automatyczne pobieranie wyników wyścigów, kwalifikacji i sprintów od roku 2018 do danych aktualnych wykorzystując `fastf1`.
- **Inżynieria cech (Feature Engineering):** Obliczanie średniej formy zespołów i kierowców, skuteczności na konkretnym torze (Track Affinity) oraz uwzględnianie punktów ze sprintów. System inteligentnie radzi sobie z nieukończonymi wyścigami (DNF), rozróżniając awarie mechaniczne od wypadków.
- **Sposób modelowania:** **Klasyfikacja** Przewiduje prawdopodobieństwo ukończenia wyścigu w pierwszej piątce (Top 5) przy użyciu `XGBClassifier`.
- **Predykcje na żywo (`live_prediction.py`):** Automatycznie identyfikuje nadchodzący wyścig, pobiera wyniki kwalifikacji (i sprintu) zaraz po ich zakończeniu i w czasie rzeczywistym przewiduje wyniki niedzielnego wyścigu.

### Struktura projektu
- `data_loader.py` - Pobiera surowe dane z API FastF1.
- `preprocessing.py` - Czyści dane, kategoryzuje statusy ukończenia (np. Awaria vs Wypadek) i uzupełnia brakujące wartości.
- `feature_engineering.py` - Wylicza średnie kroczące, historyczne osiągi na danym torze i aktualną formę.
- `model.py` - Odpowiada za podział danych (z zachowaniem chronologii), trening modeli, strojenie hiperparametrów (`GridSearchCV`) i ewaluację.
- `main.py` - Skrypty uruchomieniowe do testowania modeli na danych historycznych.
- `live_prediction.py` - Główny skrypt do predykcji na żywo w trakcie trwania weekendu wyścigowego.

### Wymagania
Zainstaluj wymagane pakiety Python:
```bash
pip install -r requirements.txt
```

### Jak uruchomić
Aby wytrenować model klasyfikacji i wygenerować predykcję dla ostatniego wyścigu z bazy historycznej:
```bash
python main.py
```
Aby uruchomić predykcję na żywo dla nadchodzącego wyścigu (najlepiej uruchomić po zakończeniu sesji kwalifikacyjnej/sprintu):
```bash
python live_prediction.py
```
