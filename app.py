import pandas as pd
import joblib
import os
import subprocess
import sys
from flask import Flask, request, jsonify, make_response

# Inicializar la aplicación Flask
app = Flask(__name__)

def preprocess_data(df: pd.DataFrame, features: list) -> (pd.DataFrame, pd.DataFrame):
    """Preprocesses the raw player data to match the model's training format."""
    # Filtrar jugadores con al menos 500 minutos
    filtered_players = df[df['MIN'] >= 500].copy()

    # Seleccionar características y normalizar por 36 minutos
    stats_cols = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FGA', 'FG_PCT', 'FG3A', 'FG3_PCT', 'FTA', 'FT_PCT', 'TOV']
    
    model_df = filtered_players[['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION', 'MIN'] + stats_cols].copy()

    for col in stats_cols:
        if col not in ['FG_PCT', 'FG3_PCT', 'FT_PCT']:
            model_df[col + '_PER36'] = (model_df[col] / model_df['MIN']) * 36

    model_df.fillna(0, inplace=True)
    
    # Asegurarse de que el DataFrame tiene todas las columnas necesarias
    for feature in features:
        if feature not in model_df.columns:
            model_df[feature] = 0

    X = model_df[features]
    
    return model_df, X

@app.route('/cluster_players/', methods=['GET'])
def cluster_players():
    """
    Endpoint para obtener clusters de jugadores de la NBA.
    Utiliza el parámetro 'k' de la URL (ej: /cluster_players/?k=7).
    """
    # --- Obtener y validar el parámetro 'k' ---
    k = request.args.get('k', default=6, type=int)
    if k <= 1:
        return make_response(jsonify(error="El parámetro 'k' debe ser un entero mayor que 1."), 400)

    model_filename = f"player_cluster_model_k{k}.joblib"

    # --- 1. Generar modelo si no existe ---
    if not os.path.exists(model_filename):
        print(f"Modelo para k={k} no encontrado. Generándolo ahora...")
        try:
            process = subprocess.run(
                [sys.executable, "generate_model.py", str(k)],
                capture_output=True, text=True, check=True, timeout=300
            )
            print(process.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error al generar el modelo: {e.stderr}")
            return make_response(jsonify(error=f"Error al generar el modelo: {e.stderr}"), 500)
        except subprocess.TimeoutExpired:
            return make_response(jsonify(error="La generación del modelo tardó demasiado tiempo."), 500)

    # --- 2. Cargar el modelo ---
    try:
        models = joblib.load(model_filename)
        scaler = models['scaler']
        kmeans = models['kmeans']
        pca = models['pca']
        features = models['features']
    except Exception as e:
        return make_response(jsonify(error=f"Error al cargar el modelo: {e}"), 500)

    # --- 3. Obtener y procesar datos frescos ---
    try:
        # Importar aquí para evitar hacerlo en cada request si no es necesario
        from nba_api.stats.endpoints import leaguedashplayerstats
        fresh_player_stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season='2023-24', 
            season_type_all_star='Regular Season'
        ).get_data_frames()[0]
    except Exception as e:
        return make_response(jsonify(error=f"Error al obtener datos frescos de la NBA: {e}"), 500)

    processed_df, X_fresh = preprocess_data(fresh_player_stats, features)
    if X_fresh.empty:
        return jsonify(message="No hay suficientes datos de jugadores para procesar.")

    # --- 4. Aplicar modelos ---
    X_scaled = scaler.transform(X_fresh)
    clusters = kmeans.predict(X_scaled)
    pca_coords = pca.transform(X_scaled)

    # --- 5. Formatear y devolver resultados ---
    processed_df['cluster'] = clusters
    processed_df['pc1'] = pca_coords[:, 0]
    processed_df['pc2'] = pca_coords[:, 1]
    
    result_cols = ['PLAYER_NAME', 'TEAM_ABBREVIATION', 'cluster', 'pc1', 'pc2']
    results = processed_df[result_cols].to_dict(orient='records')
    
    return jsonify(k=k, player_clusters=results)

if __name__ == '__main__':
    app.run(debug=True, port=8000)