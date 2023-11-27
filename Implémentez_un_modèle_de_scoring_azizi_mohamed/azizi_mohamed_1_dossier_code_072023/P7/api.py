import pandas as pd
from flask import Flask, request, jsonify
import joblib
import shap
from catboost import CatBoostClassifier, Pool
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import numpy as np 
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

app = Flask(__name__)

# Charger le modèle CatBoost pré-entraîné
model_path = '/Users/azizi/Desktop/appli3/modele_catboost.pkl'
pipeline = joblib.load(model_path)
model = pipeline.named_steps['model']

# Charger les données de test
data_test_path = '/Users/azizi/Desktop/appli3/X_test.csv'
data_test = pd.read_csv(data_test_path)
# Supprimer les lignes contenant des valeurs NaN
data_test = data_test.dropna()

# Route pour effectuer des prédictions pour un client donné
@app.route('/api/predict/<int:SK_ID_CURR>', methods=['GET'])
def predict(SK_ID_CURR):
    client_data = data_test[data_test['SK_ID_CURR'] == SK_ID_CURR]
    prediction = model.predict(client_data)
    return jsonify({'SK_ID_CURR': SK_ID_CURR, 'prediction': int(prediction[0])})

# Route pour obtenir la capacité de remboursement pour un client donné
@app.route('/api/capacite-remboursement/<int:SK_ID_CURR>', methods=['GET'])
def get_capacite_remboursement(SK_ID_CURR):
    client_data = data_test[data_test['SK_ID_CURR'] == SK_ID_CURR]
    proba_remboursement = model.predict_proba(client_data)[:, 1]
    return jsonify({'SK_ID_CURR': SK_ID_CURR, 'capacite_remboursement': float(proba_remboursement[0])})

# Route pour la page d'accueil
@app.route('/', methods=['GET'])
def home():
    return 'Bienvenue sur l\'API de prédiction de remboursement des clients.'

# Nouvelle route pour obtenir l'importance des caractéristiques du modèle pour un client donné
@app.route('/api/feature-importance/<int:SK_ID_CURR>', methods=['GET'])
def get_feature_importance(SK_ID_CURR):
    client_data = data_test[data_test['SK_ID_CURR'] == SK_ID_CURR]

    # Calculer l'importance des caractéristiques à partir du modèle (exemple avec CatBoost)
    feature_importances = model.feature_importances_

    # Créez un dictionnaire avec les importances des caractéristiques
    feature_names = client_data.columns.tolist()
    feature_importance_dict = {feature_names[i]: feature_importances[i] for i in range(len(feature_names))}

    return jsonify({'SK_ID_CURR': SK_ID_CURR, 'feature_importance': feature_importance_dict})

@app.route('/api/similar-clients/<int:SK_ID_CURR>', methods=['GET'])
def get_similar_clients(SK_ID_CURR):
    client_data = data_test[data_test['SK_ID_CURR'] == SK_ID_CURR]
    
    # Récupérez les noms de colonnes de data_test
    feature_columns = data_test.columns.tolist()
    
    # Excluez la colonne 'SK_ID_CURR' s'il existe dans data_test
    if 'SK_ID_CURR' in feature_columns:
        feature_columns.remove('SK_ID_CURR')
    
    # Remplacez les valeurs NaN par la moyenne des colonnes
    imputer = SimpleImputer(strategy="mean")
    data_test_imputed = imputer.fit_transform(data_test[feature_columns])
    
    client_features = client_data[feature_columns]

    # Utilisez K-Means pour trouver des clients similaires
    kmeans = KMeans(n_clusters=100, random_state=0)
    kmeans.fit(data_test_imputed)

    # Obtenir les étiquettes de cluster du client donné
    client_cluster_label = kmeans.predict(client_features)

    # Trouver les clients dans le même cluster que le client donné
    similar_client_indices = np.where(kmeans.labels_ == client_cluster_label[0])

    # Exclure le client donné de la liste des clients similaires
    similar_client_indices = [i for i in similar_client_indices[0] if i != SK_ID_CURR]

    # Obtenir les informations des 5 clients les plus similaires
    similar_clients_info = data_test.iloc[similar_client_indices[:5]]

    # Convertir les informations en un dictionnaire
    similar_clients_dict = similar_clients_info.to_dict(orient='records')

    return jsonify({'SK_ID_CURR': SK_ID_CURR, 'similar_clients': similar_clients_dict})

if __name__ == '__main__':
    app.run(debug=True)