import pandas as pd
import joblib
import os
import subprocess
import sys
import time
import copy
import numpy as np
from flask import Flask, request, jsonify, make_response

# Imports from notebook
from nba_api.stats.endpoints import leaguegamefinder, leaguedashplayerstats
from nba_api.stats.static import teams
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

import xgboost as xgb


# ==============================================================================
# SETUP ON STARTUP: LOAD MODELS AND PREPARE DATA
# ==============================================================================

# --- Helper Functions from Notebook (for XGBoost model) ---

def seasons_from_start_years(start_year=2015, end_year_inclusive=2025):
    start_years = range(start_year, end_year_inclusive)
    return [f"{y}-{str(y + 1)[-2:]}" for y in start_years]


def fetch_official_nba_regular_season(start_year=2015, end_year=2025, sleep_s=0.6):
    all_dfs = []
    for season in seasons_from_start_years(start_year, end_year):
        print(f"Fetching data for season: {season}...")
        gf = leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            league_id_nullable="00",
            season_type_nullable="Regular Season"
        )
        df_season = gf.get_data_frames()[0].copy()
        df_season["SEASON_STR"] = season
        df_season["SEASON_TYPE"] = "Regular Season"
        all_dfs.append(df_season)
        time.sleep(sleep_s)
    df = pd.concat(all_dfs, ignore_index=True)
    nba_team_ids = {t["id"] for t in teams.get_teams()}
    df = df[df["TEAM_ID"].isin(nba_team_ids)].copy()
    return df


def build_team_game_table_from_raw(df_raw):
    df = df_raw.copy()
    keep = [
        "SEASON_ID", "GAME_ID", "GAME_DATE", "TEAM_ID", "TEAM_ABBREVIATION",
        "TEAM_NAME", "MATCHUP", "WL", "MIN", "PTS",
        "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
        "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF",
        "PLUS_MINUS", "SEASON_STR", "SEASON_TYPE"
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    df = df[df["GAME_DATE"].notna()].copy()
    df["WIN"] = (df["WL"] == "W").astype(int)
    df["IS_HOME"] = df["MATCHUP"].str.contains("vs.", na=False, regex=False).astype(int)
    df = df.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    df["DAYS_REST"] = df.groupby("TEAM_ID")["GAME_DATE"].diff().dt.days
    df["DAYS_REST"] = df["DAYS_REST"].fillna(df["DAYS_REST"].median()).astype(int)
    return df


def add_pregame_features_teamgame(df, window=5, min_periods=None, keep_early_games=True):
    df = df.copy()
    df = df.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    if min_periods is None:
        min_periods = window
    base_cols = ["PTS", "FG_PCT", "FG3_PCT", "REB", "AST", "TOV", "OREB", "FGA", "FTA", "PLUS_MINUS"]
    base_cols = [c for c in base_cols if c in df.columns]
    for c in base_cols:
        df[f"{c}_roll{window}"] = (
            df.groupby("TEAM_ID")[c]
            .transform(lambda s: s.rolling(window=window, min_periods=min_periods).mean().shift(1))
        )
    std_cols = [c for c in ["PTS", "FG_PCT", "FG3_PCT"] if c in df.columns]
    for c in std_cols:
        df[f"{c}_std{window}"] = (
            df.groupby("TEAM_ID")[c]
            .transform(lambda s: s.rolling(window=window, min_periods=min_periods).std().shift(1))
        )
    df[f"WINRATE_last{window}"] = (
        df.groupby("TEAM_ID")["WIN"]
        .transform(lambda s: s.rolling(window=window, min_periods=min_periods).mean().shift(1))
    )
    prev_win = df.groupby("TEAM_ID")["WIN"].shift(1)
    cut = (prev_win != 1) | (prev_win.isna())
    run_id = cut.groupby(df["TEAM_ID"]).cumsum()
    df["WIN_STREAK"] = prev_win.groupby([df["TEAM_ID"], run_id]).cumsum().fillna(0).astype(int)

    roll_like = [c for c in df.columns if c.endswith(f"_roll{window}")]
    core_check = roll_like + [f"WINRATE_last{window}", "WIN_STREAK", "DAYS_REST"]
    core_check = [c for c in core_check if c in df.columns]
    df["HAS_ROLL_FEATURES"] = df[core_check].notna().all(axis=1).astype(int)

    if not keep_early_games:
        df = df[df["HAS_ROLL_FEATURES"] == 1].reset_index(drop=True)
    return df


def build_single_game_features(df_team_feat: pd.DataFrame, home_abbr: str, away_abbr: str, as_of_date=None,
                               window=5) -> pd.DataFrame:
    df = df_team_feat.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    home_abbr = home_abbr.upper()
    away_abbr = away_abbr.upper()

    if as_of_date is None:
        last_date = min(
            df[df["TEAM_ABBREVIATION"] == home_abbr]["GAME_DATE"].max(),
            df[df["TEAM_ABBREVIATION"] == away_abbr]["GAME_DATE"].max()
        )
        as_of_date = last_date + pd.Timedelta(days=1)
    else:
        as_of_date = pd.to_datetime(as_of_date)

    home_hist = df[(df["TEAM_ABBREVIATION"] == home_abbr) & (df["GAME_DATE"] < as_of_date)].sort_values("GAME_DATE")
    away_hist = df[(df["TEAM_ABBREVIATION"] == away_abbr) & (df["GAME_DATE"] < as_of_date)].sort_values("GAME_DATE")

    if home_hist.empty or away_hist.empty:
        raise ValueError("No hay historial suficiente para uno de los equipos antes de esa fecha.")

    home_row = home_hist.iloc[-1].copy()
    away_row = away_hist.iloc[-1].copy()

    if int(home_row.get("HAS_ROLL_FEATURES", 0)) != 1 or int(away_row.get("HAS_ROLL_FEATURES", 0)) != 1:
        raise ValueError("Uno de los equipos no tiene features pregame completas (HAS_ROLL_FEATURES=0).")

    out = {}
    out["H_DAYS_REST"] = (as_of_date - home_row['GAME_DATE']).days
    out["A_DAYS_REST"] = (as_of_date - away_row['GAME_DATE']).days
    out["LAL_B2B"] = 1 if out["H_DAYS_REST"] <= 1 else 0  # Assuming 'LAL' was a placeholder for 'HOME'

    roll_cols = [c for c in df.columns if c.endswith(f"_roll{window}")]
    std_cols = [c for c in df.columns if c.endswith(f"_std{window}")]
    extra_cols = ["WIN_STREAK", f"WINRATE_last{window}"]
    cols_to_use = [c for c in (roll_cols + std_cols + extra_cols) if c in df.columns]

    for c in cols_to_use:
        out[f"H_{c}"] = home_row[c]
        out[f"A_{c}"] = away_row[c]

    # Differentials
    out["DIFF_DAYS_REST"] = out["H_DAYS_REST"] - out["A_DAYS_REST"]
    for c in roll_cols + std_cols + extra_cols:
        h, a = f"H_{c}", f"A_{c}"
        if h in out and a in out:
            out[f"DIFF_{c}"] = out[h] - out[a]

    # Add ES_CASA (always 1 for this endpoint)
    out["ES_CASA"] = 1

    return pd.DataFrame([out])


print("--- Initializing Application ---")
# Global variables for models and data
xgb_model = None
xgb_feature_cols = None
xgb_base_data = None
playoff_model = None

try:
    # 1. Load the XGBoost model bundle
    print("Loading XGBoost model from joblib...")
    xgb_bundle = joblib.load("xgb_isotonic_nba_homewin.joblib")
    xgb_model = xgb_bundle["model_calibrated"]
    xgb_feature_cols = xgb_bundle["feature_cols"]
    print("XGBoost model loaded successfully.")

    # 2. Load the Playoff prediction model
    print("Loading Playoff prediction model from joblib...")
    playoff_model = joblib.load("modelo_nba.joblib")
    print("Playoff prediction model loaded successfully.")

    # 3. Prepare the base data required for feature generation
    print("Fetching and preparing base NBA data (2015-2025)...")
    raw_data = fetch_official_nba_regular_season(start_year=2015, end_year=2025)
    team_game_data = build_team_game_table_from_raw(raw_data)
    xgb_base_data = add_pregame_features_teamgame(team_game_data, window=5, keep_early_games=True)
    print("Base NBA data is ready.")

except Exception as e:
    print(f"FATAL: Could not initialize models or data on startup: {e}", file=sys.stderr)
    xgb_model = None  # Ensure model is None if setup fails
    playoff_model = None

print("--- Application Ready ---")

# Inicializar la aplicación Flask
app = Flask(__name__)


# --- Helper function for K-Means endpoint ---
def preprocess_data_kmeans(df: pd.DataFrame, features: list) -> (pd.DataFrame, pd.DataFrame):
    """Preprocesses the raw player data for the K-Means model."""
    filtered_players = df[df['MIN'] >= 500].copy()
    stats_cols = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FGA', 'FG_PCT', 'FG3A', 'FG3_PCT', 'FTA', 'FT_PCT', 'TOV']
    model_df = filtered_players[['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION', 'MIN'] + stats_cols].copy()
    for col in stats_cols:
        if col not in ['FG_PCT', 'FG3_PCT', 'FT_PCT']:
            model_df[col + '_PER36'] = (model_df[col] / model_df['MIN']) * 36
    model_df.fillna(0, inplace=True)
    for feature in features:
        if feature not in model_df.columns:
            model_df[feature] = 0
    X = model_df[features]
    return model_df, X


# ==============================================================================
# ENDPOINTS
# ==============================================================================
# Endpoint para clustering de jugadores (David Ceballos)
@app.route('/cluster_players/', methods=['GET'])
def cluster_players():
    """Endpoint to get clusters of NBA players."""
    k = request.args.get('k', default=6, type=int)
    if k <= 1:
        return make_response(jsonify(error="El parámetro 'k' debe ser un entero mayor que 1."), 400)

    model_filename = f"player_cluster_model_k{k}.joblib"

    if not os.path.exists(model_filename):
        return make_response(jsonify(error=f"Modelo para k={k} no encontrado."), 404)

    try:
        models = joblib.load(model_filename)
        scaler, kmeans, pca, features = models['scaler'], models['kmeans'], models['pca'], models['features']
    except Exception as e:
        return make_response(jsonify(error=f"Error al cargar el modelo: {e}"), 500)

    try:
        fresh_player_stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season='2023-24', season_type_all_star='Regular Season'
        ).get_data_frames()[0]
    except Exception as e:
        return make_response(jsonify(error=f"Error al obtener datos frescos de la NBA: {e}"), 500)

    processed_df, X_fresh = preprocess_data_kmeans(fresh_player_stats, features)
    if X_fresh.empty:
        return jsonify(message="No hay suficientes datos de jugadores para procesar.")

    X_scaled = scaler.transform(X_fresh)
    clusters = kmeans.predict(X_scaled)
    pca_coords = pca.transform(X_scaled)

    processed_df['cluster'] = clusters
    processed_df['pc1'] = pca_coords[:, 0]
    processed_df['pc2'] = pca_coords[:, 1]

    result_cols = ['PLAYER_NAME', 'TEAM_ABBREVIATION', 'cluster', 'pc1', 'pc2']
    results = processed_df[result_cols].to_dict(orient='records')

    return jsonify(k=k, player_clusters=results)

# Endpoint para predicción de partidos NBA (XGBoost Luis Arias)
@app.route('/predict_matchup', methods=['POST'])
def predict_matchup():
    """Endpoint to predict the outcome of a single NBA matchup."""
    if not xgb_model:
        return make_response(jsonify(error="El modelo de predicción de partidos no está inicializado."), 503)

    json_data = request.get_json()
    if not json_data:
        return make_response(jsonify(error="Request body debe ser JSON."), 400)

    home_abbr = json_data.get('home_abbr')
    away_abbr = json_data.get('away_abbr')
    as_of_date = json_data.get('as_of_date')  # Opcional

    if not home_abbr or not away_abbr:
        return make_response(jsonify(error="Se requieren 'home_abbr' y 'away_abbr' en el JSON."), 400)

    try:
        feature_row = build_single_game_features(
            df_team_feat=xgb_base_data,
            home_abbr=home_abbr,
            away_abbr=away_abbr,
            as_of_date=as_of_date,
            window=5
        )
        X_one = feature_row.reindex(columns=xgb_feature_cols)

        if X_one.isna().any().any():
            missing_cols = X_one.columns[X_one.isna().any()].tolist()
            return make_response(jsonify(error=f"No se pudieron generar todas las features. Faltan datos para: {missing_cols}"), 500)

        p_home = float(xgb_model.predict_proba(X_one)[:, 1][0])
        p_away = 1.0 - p_home

        response = {
            "home_team": home_abbr,
            "away_team": away_abbr,
            "prediction_date": as_of_date if as_of_date else "latest",
            "home_win_probability": round(p_home, 4),
            "away_win_probability": round(p_away, 4)
        }
        return jsonify(response)

    except ValueError as e:
        return make_response(jsonify(error=str(e)), 400)
    except Exception as e:
        print(f"ERROR: /predict_matchup failed: {e}", file=sys.stderr)
        return make_response(jsonify(error="Ocurrió un error interno al procesar la predicción."), 500)

# Endpoint para predicción de playoffs NBA (Rubén Flores)
@app.route('/predict_playoffs', methods=['POST'])
def predict_playoffs():
    """Endpoint to predict if a team will make the playoffs based on its stats."""
    if not playoff_model:
        return make_response(jsonify(error="El modelo de predicción de playoffs no está inicializado."), 503)

    json_data = request.get_json()
    if not json_data:
        return make_response(jsonify(error="Request body debe ser JSON."), 400)

    # Validate required features
    required_features = ['WinPCT', 'PointsPG', 'OppPointsPG', 'DiffPointsPG']
    if not all(feat in json_data for feat in required_features):
        return make_response(jsonify(error=f"Faltan features. Se requieren: {required_features}"), 400)

    try:
        # Create DataFrame from input
        input_data = {feat: [json_data[feat]] for feat in required_features}
        input_df = pd.DataFrame(input_data)

        # Predict outcome and probability
        prediction = int(playoff_model.predict(input_df)[0])
        probabilities = playoff_model.predict_proba(input_df)[0]
        
        prob_no_playoffs = round(probabilities[0], 4)
        prob_playoffs = round(probabilities[1], 4)

        response = {
            "input_stats": json_data,
            "prediction_label": "Hace Playoffs" if prediction == 1 else "No Hace Playoffs",
            "prediction_value": prediction,
            "probability_no_playoffs": prob_no_playoffs,
            "probability_playoffs": prob_playoffs
        }
        return jsonify(response)
        
    except (TypeError, ValueError):
         return make_response(jsonify(error="Todas las features deben ser valores numéricos."), 400)
    except Exception as e:
        print(f"ERROR: /predict_playoffs failed: {e}", file=sys.stderr)
        return make_response(jsonify(error="Ocurrió un error interno al procesar la predicción."), 500)


if __name__ == '__main__':
    app.run(debug=False, port=8000)