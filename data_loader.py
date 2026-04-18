import fastf1
import os
import pandas as pd
import time

# Włączamy cache - to krytyczne przy FastF1, aby nie obciążać serwerów API 
# i nie pobierać gigabajtów danych telemetrycznych przy każdym uruchomieniu skryptu.
cache_dir = 'cache_folder'
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

# Tworzymy folder na wyeksportowane pliki CSV, żeby nie przetwarzać danych dwukrotnie
csv_dir = 'dane_csv'
os.makedirs(csv_dir, exist_ok=True)

def pobierz_wyniki_sezonu(rok):
    plik_wyscigi = f'{csv_dir}/wyscigi_{rok}.csv'
    plik_kwalifikacje = f'{csv_dir}/kwalifikacje_{rok}.csv'
    biezacy_rok = pd.Timestamp.now().year
    
    # Sprawdzenie, czy pobraliśmy już te dane w przeszłości.
    # Robimy to TYLKO dla zakończonych sezonów, by nie "zamrozić" trwającego roku.
    if rok < biezacy_rok and os.path.exists(plik_wyscigi) and os.path.exists(plik_kwalifikacje):
        print(f"Dane dla sezonu {rok} wczytano szybko z lokalnych plików CSV.")
        return pd.read_csv(plik_wyscigi), pd.read_csv(plik_kwalifikacje)
        
    print(f"Pobieranie danych dla sezonu {rok} z API FastF1...")
    harmonogram = fastf1.get_event_schedule(rok)
    wyniki_wyscigow = []
    wyniki_kwalifikacji = []
    sukces_sezonu = True
    
    for _, event in harmonogram.iterrows():
        # Pomijamy przedsezonowe testy i wyścigi, które się jeszcze nie odbyły
        if event['EventFormat'] == 'testing' or event['EventDate'] > pd.Timestamp.now(tz=event['EventDate'].tz):
            continue
            
        try:
            # Pobieranie wyścigu ('R')
            sesja_r = fastf1.get_session(rok, event['EventName'], 'R')
            sesja_r.load(telemetry=False, weather=False, messages=False)
            df_r = sesja_r.results.copy()  # Używamy .copy() aby uniknąć SettingWithCopyWarning
            df_r['Year'] = rok
            df_r['EventName'] = event['EventName']
            wyniki_wyscigow.append(df_r)
            
            # Pobieranie kwalifikacji ('Q')
            sesja_q = fastf1.get_session(rok, event['EventName'], 'Q')
            sesja_q.load(telemetry=False, weather=False, messages=False)
            df_q = sesja_q.results.copy()
            df_q['Year'] = rok
            df_q['EventName'] = event['EventName']
            wyniki_kwalifikacji.append(df_q)
            
        except Exception as e:
            print(f"Błąd podczas pobierania {event['EventName']} {rok}: {e}")
            # Jeśli to limit API, przerywamy pętlę dla tego sezonu, żeby nie zapisać pustych danych
            if "calls/h" in str(e) or "limit" in str(e).lower():
                print("-> Osiągnięto limit API! Przerwano pobieranie reszty wyścigów w tym sezonie.")
                sukces_sezonu = False
                break
                
        # Małe opóźnienie, żeby być bardziej "przyjaznym" dla API
        time.sleep(0.5)
            
    df_w = pd.concat(wyniki_wyscigow, ignore_index=True) if wyniki_wyscigow else pd.DataFrame()
    df_k = pd.concat(wyniki_kwalifikacji, ignore_index=True) if wyniki_kwalifikacji else pd.DataFrame()
    
    # Zapisujemy do CSV TYLKO zakończone sezony na przyszłość
    if sukces_sezonu and not df_w.empty and not df_k.empty and rok < biezacy_rok:
        df_w.to_csv(plik_wyscigi, index=False)
        df_k.to_csv(plik_kwalifikacje, index=False)
    elif not sukces_sezonu:
        print(f"-> UWAGA: Nie zapisano CSV dla {rok} z powodu limitu API. Uruchom skrypt ponownie później.")
        
    return df_w, df_k

hist_wyscigi = []
hist_kwalifikacje = []
for rok in [2022, 2023, 2024, 2025]:
    df_w, df_k = pobierz_wyniki_sezonu(rok)
    hist_wyscigi.append(df_w)
    hist_kwalifikacje.append(df_k)

df_historyczne_wyscigi = pd.concat(hist_wyscigi, ignore_index=True)
df_historyczne_kwalifikacje = pd.concat(hist_kwalifikacje, ignore_index=True)

df_aktualne_wyscigi, df_aktualne_kwalifikacje = pobierz_wyniki_sezonu(2026)

print(f"\nSezony historyczne -> Wyścigi: {len(df_historyczne_wyscigi)} rekordów | Kwalifikacje: {len(df_historyczne_kwalifikacje)} rekordów.")
print(f"Sezon aktualny     -> Wyścigi: {len(df_aktualne_wyscigi)} rekordów | Kwalifikacje: {len(df_aktualne_kwalifikacje)} rekordów.")
