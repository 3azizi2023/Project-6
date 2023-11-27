from sklearn import model_selection
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import requests
import joblib
import shap

api_url_predict = 'http://localhost:5000/api/predict/'

# Charger le modèle CatBoost pré-entraîné
model_path = '/Users/azizi/Desktop/Implémentez_un_modèle_de_scoring_azizi_mohamed/azizi_mohamed_1_dossier_code_072023/modele_catboost.pkl' 
pipeline = joblib.load(model_path)

# Extrait le modèle CatBoost du pipeline
model = pipeline.named_steps['model']

# Charger les données de test à partir de X_test.csv
X_test = pd.read_csv('/Users/azizi/Desktop/Implémentez_un_modèle_de_scoring_azizi_mohamed/azizi_mohamed_1_dossier_code_072023/X_test.csv')
X_test = X_test.dropna()

# Dictionnaire des noms de variables avec des descriptions
variable_names = {
    'SK_ID_CURR': 'SK_ID_CURR : Identifiable unique du client',
    'TARGET': 'TARGET : Défaut de paiement (0: normal, 1: défaut)',
    'NAME_CONTRACT_TYPE': 'NAME_CONTRACT_TYPE : Type de contrat de prêt',
    'CODE_GENDER': 'CODE_GENDER : Genre du client',
    'FLAG_OWN_CAR': 'FLAG_OWN_CAR : Possède une voiture',
    'FLAG_OWN_REALTY': 'FLAG_OWN_REALTY : Possède un bien immobilier',
    'CNT_CHILDREN': 'CNT_CHILDREN : Nombre d\'enfants à charge du client',
    'AMT_INCOME_TOTAL': 'AMT_INCOME_TOTAL : Revenu total du client',
    'AMT_CREDIT': 'AMT_CREDIT : Montant total du crédit demandé par le client',
    'AMT_ANNUITY': 'AMT_ANNUITY : Montant de l\'annuité du prêt',
    'AMT_GOODS_PRICE': 'AMT_GOODS_PRICE : Prix des biens pour lesquels le prêt a été demandé',
    'NAME_TYPE_SUITE': 'NAME_TYPE_SUITE : Type de personne accompagnant le client lors de la demande de prêt',
    'NAME_INCOME_TYPE': 'NAME_INCOME_TYPE : Source de revenu du client',
    'NAME_EDUCATION_TYPE': 'NAME_EDUCATION_TYPE : Niveau d\'éducation du client',
    'NAME_FAMILY_STATUS': 'NAME_FAMILY_STATUS : Statut familial du client',
    'NAME_HOUSING_TYPE': 'NAME_HOUSING_TYPE : Type de logement du client',
    'REGION_POPULATION_RELATIVE': 'REGION_POPULATION_RELATIVE : Population relative de la région où vit le client',
    'DAYS_BIRTH': 'DAYS_BIRTH : Nombre de jours depuis la naissance du client',
    'DAYS_EMPLOYED': 'DAYS_EMPLOYED : Nombre de jours depuis que le client est employé',
    'DAYS_REGISTRATION': 'DAYS_REGISTRATION : Nombre de jours depuis l\'enregistrement du client dans la base de données',
    'DAYS_ID_PUBLISH': 'DAYS_ID_PUBLISH : Nombre de jours depuis la publication de l\'identifiant du client',
    'OWN_CAR_AGE': 'OWN_CAR_AGE : Âge de la voiture du client',
    'FLAG_MOBIL': 'FLAG_MOBIL : Possède un téléphone mobile',
    'FLAG_EMP_PHONE': 'FLAG_EMP_PHONE : Possède un téléphone professionnel',
    'FLAG_WORK_PHONE': 'FLAG_WORK_PHONE : Possède un téléphone professionnel',
    'FLAG_CONT_MOBILE': 'FLAG_CONT_MOBILE : Peut être contacté sur téléphone mobile',
    'FLAG_PHONE': 'FLAG_PHONE : Possède un téléphone',
    'FLAG_EMAIL': 'FLAG_EMAIL : Possède un e-mail',
    'OCCUPATION_TYPE': 'OCCUPATION_TYPE : Type d\'occupation du client',
    'CNT_FAM_MEMBERS': 'CNT_FAM_MEMBERS : Nombre de membres de la famille du client',
    'REGION_RATING_CLIENT': 'REGION_RATING_CLIENT : Note de la région où vit le client',
    'REGION_RATING_CLIENT_W_CITY': 'REGION_RATING_CLIENT_W_CITY : Note de la région où vit le client avec correction de la ville',
    'WEEKDAY_APPR_PROCESS_START': 'WEEKDAY_APPR_PROCESS_START : Jour de la semaine où la demande de prêt a été soumise',
    'HOUR_APPR_PROCESS_START': 'HOUR_APPR_PROCESS_START : Heure à laquelle la demande de prêt a été soumise',
    'REG_REGION_NOT_LIVE_REGION': 'REG_REGION_NOT_LIVE_REGION : Région différente de celle de l\'adresse actuelle (client ne vit pas dans la région de l\'adresse actuelle)',
    'REG_REGION_NOT_WORK_REGION': 'REG_REGION_NOT_WORK_REGION : Région différente de celle du travail actuel (client ne travaille pas dans la région de l\'adresse actuelle)',
    'LIVE_REGION_NOT_WORK_REGION': 'LIVE_REGION_NOT_WORK_REGION : Région différente de celle du travail actuel (client ne travaille pas dans la région de l\'adresse actuelle)',
    'REG_CITY_NOT_LIVE_CITY': 'REG_CITY_NOT_LIVE_CITY : Ville différente de celle de l\'adresse actuelle (client ne vit pas dans la ville de l\'adresse actuelle)',
    'REG_CITY_NOT_WORK_CITY': 'REG_CITY_NOT_WORK_CITY : Ville différente de celle du travail actuel (client ne travaille pas dans la ville de l\'adresse actuelle)',
    'LIVE_CITY_NOT_WORK_CITY': 'LIVE_CITY_NOT_WORK_CITY : Ville différente de celle du travail actuel (client ne travaille pas dans la ville de l\'adresse actuelle)',
    'ORGANIZATION_TYPE': 'ORGANIZATION_TYPE : Type d\'organisation où travaille le client',
    'EXT_SOURCE_1': 'EXT_SOURCE_1 : Indice de Solvabilité - Source Externe 1',
    'EXT_SOURCE_2': 'EXT_SOURCE_2 : Indice de Solvabilité - Source Externe 2',
    'EXT_SOURCE_3': 'EXT_SOURCE_3 : Indice de Solvabilité - Source Externe 3',
    'APARTMENTS_AVG': 'APARTMENTS_AVG : Moyenne des appartements dans le bâtiment',
    'BASEMENTAREA_AVG': 'BASEMENTAREA_AVG : Moyenne de la superficie du sous-sol dans le bâtiment',
    'YEARS_BEGINEXPLUATATION_AVG': 'YEARS_BEGINEXPLUATATION_AVG : Moyenne des années d\'exploitation du bâtiment',
    'ELEVATORS_AVG': 'ELEVATORS_AVG : Moyenne du nombre d\'ascenseurs dans le bâtiment',
    'ENTRANCES_AVG': 'ENTRANCES_AVG : Moyenne du nombre d\'entrées dans le bâtiment',
    'FLOORSMAX_AVG': 'FLOORSMAX_AVG : Moyenne du nombre d\'étages maximum dans le bâtiment',
    'LANDAREA_AVG': 'LANDAREA_AVG : Moyenne de la superficie du terrain autour du bâtiment',
    'LIVINGAREA_AVG': 'LIVINGAREA_AVG : Moyenne de la superficie habitable dans le bâtiment',
    'NONLIVINGAREA_AVG': 'NONLIVINGAREA_AVG : Moyenne de la superficie non habitable dans le bâtiment',
    'APARTMENTS_MODE': 'APARTMENTS_MODE : Mode des appartements dans le bâtiment',
    'BASEMENTAREA_MODE': 'BASEMENTAREA_MODE : Mode de la superficie du sous-sol dans le bâtiment',
    'YEARS_BEGINEXPLUATATION_MODE': 'YEARS_BEGINEXPLUATATION_MODE : Mode des années d\'exploitation du bâtiment',
    'ELEVATORS_MODE': 'ELEVATORS_MODE : Mode du nombre d\'ascenseurs dans le bâtiment',
    'ENTRANCES_MODE': 'ENTRANCES_MODE : Mode du nombre d\'entrées dans le bâtiment',
    'FLOORSMAX_MODE': 'FLOORSMAX_MODE : Mode du nombre d\'étages maximum dans le bâtiment',
    'LANDAREA_MODE': 'LANDAREA_MODE : Mode de la superficie du terrain autour du bâtiment',
    'LIVINGAREA_MODE': 'LIVINGAREA_MODE : Mode de la superficie habitable dans le bâtiment',
    'NONLIVINGAREA_MODE': 'NONLIVINGAREA_MODE : Mode de la superficie non habitable dans le bâtiment',
    'APARTMENTS_MEDI': 'APARTMENTS_MEDI : Médiane des appartements dans le bâtiment',
    'BASEMENTAREA_MEDI': 'BASEMENTAREA_MEDI : Médiane de la superficie du sous-sol dans le bâtiment',
    'YEARS_BEGINEXPLUATATION_MEDI': 'YEARS_BEGINEXPLUATATION_MEDI : Médiane des années d\'exploitation du bâtiment',
    'ELEVATORS_MEDI': 'ELEVATORS_MEDI : Médiane du nombre d\'ascenseurs dans le bâtiment',
    'ENTRANCES_MEDI': 'ENTRANCES_MEDI : Médiane du nombre d\'entrées dans le bâtiment',
    'FLOORSMAX_MEDI': 'FLOORSMAX_MEDI : Médiane du nombre d\'étages maximum dans le bâtiment',
    'LANDAREA_MEDI': 'LANDAREA_MEDI : Médiane de la superficie du terrain autour du bâtiment',
    'LIVINGAREA_MEDI': 'LIVINGAREA_MEDI : Médiane de la superficie habitable dans le bâtiment',
    'NONLIVINGAREA_MEDI': 'NONLIVINGAREA_MEDI : Médiane de la superficie non habitable dans le bâtiment',
    'FONDKAPREMONT_MODE': 'FONDKAPREMONT_MODE : Mode du fonds de réserve du bâtiment',
    'HOUSETYPE_MODE': 'HOUSETYPE_MODE : Mode du type de maison',
    'TOTALAREA_MODE': 'TOTALAREA_MODE : Mode de la superficie totale',
    'WALLSMATERIAL_MODE': 'WALLSMATERIAL_MODE : Mode du matériau des murs',
    'EMERGENCYSTATE_MODE': 'EMERGENCYSTATE_MODE : Mode de l\'état d\'urgence',
    'OBS_30_CNT_SOCIAL_CIRCLE': 'OBS_30_CNT_SOCIAL_CIRCLE : Nombre d\'observations de statut DPD (jours de retard de paiement) au delà de 30 jours dans les cercles sociaux',
    'DEF_30_CNT_SOCIAL_CIRCLE': 'DEF_30_CNT_SOCIAL_CIRCLE : Nombre de défauts de paiement au delà de 30 jours dans les cercles sociaux',
    'OBS_60_CNT_SOCIAL_CIRCLE': 'OBS_60_CNT_SOCIAL_CIRCLE : Nombre d\'observations de statut DPD (jours de retard de paiement) au delà de 60 jours dans les cercles sociaux',
    'DEF_60_CNT_SOCIAL_CIRCLE': 'DEF_60_CNT_SOCIAL_CIRCLE : Nombre de défauts de paiement au delà de 60 jours dans les cercles sociaux',
    'DAYS_LAST_PHONE_CHANGE': 'DAYS_LAST_PHONE_CHANGE : Nombre de jours depuis le dernier changement de numéro de téléphone',
    'FLAG_DOCUMENT_2': 'FLAG_DOCUMENT_2 : Indicateur si le client a fourni le document 2',
    'FLAG_DOCUMENT_3': 'FLAG_DOCUMENT_3 : Indicateur si le client a fourni le document 3',
    'FLAG_DOCUMENT_4': 'FLAG_DOCUMENT_4 : Indicateur si le client a fourni le document 4',
    'FLAG_DOCUMENT_5': 'FLAG_DOCUMENT_5 : Indicateur si le client a fourni le document 5',
    'FLAG_DOCUMENT_6': 'FLAG_DOCUMENT_6 : Indicateur si le client a fourni le document 6',
    'FLAG_DOCUMENT_7': 'FLAG_DOCUMENT_7 : Indicateur si le client a fourni le document 7',
    'FLAG_DOCUMENT_8': 'FLAG_DOCUMENT_8 : Indicateur si le client a fourni le document 8',
    'FLAG_DOCUMENT_9': 'FLAG_DOCUMENT_9 : Indicateur si le client a fourni le document 9',
    'FLAG_DOCUMENT_10': 'FLAG_DOCUMENT_10 : Indicateur si le client a fourni le document 10',
    'FLAG_DOCUMENT_11': 'FLAG_DOCUMENT_11 : Indicateur si le client a fourni le document 11',
    'FLAG_DOCUMENT_12': 'FLAG_DOCUMENT_12 : Indicateur si le client a fourni le document 12',
    'FLAG_DOCUMENT_13': 'FLAG_DOCUMENT_13 : Indicateur si le client a fourni le document 13',
    'FLAG_DOCUMENT_14': 'FLAG_DOCUMENT_14 : Indicateur si le client a fourni le document 14',
    'FLAG_DOCUMENT_15': 'FLAG_DOCUMENT_15 : Indicateur si le client a fourni le document 15',
    'FLAG_DOCUMENT_16': 'FLAG_DOCUMENT_16 : Indicateur si le client a fourni le document 16',
    'FLAG_DOCUMENT_17': 'FLAG_DOCUMENT_17 : Indicateur si le client a fourni le document 17',
    'FLAG_DOCUMENT_18': 'FLAG_DOCUMENT_18 : Indicateur si le client a fourni le document 18',
    'FLAG_DOCUMENT_19': 'FLAG_DOCUMENT_19 : Indicateur si le client a fourni le document 19',
    'FLAG_DOCUMENT_20': 'FLAG_DOCUMENT_20 : Indicateur si le client a fourni le document 20',
    'FLAG_DOCUMENT_21': 'FLAG_DOCUMENT_21 : Indicateur si le client a fourni le document 21',
    'AMT_REQ_CREDIT_BUREAU_HOUR': 'AMT_REQ_CREDIT_BUREAU_HOUR : Nombre de demandes de renseignements auprès du bureau de crédit dans la même heure',
    'AMT_REQ_CREDIT_BUREAU_DAY': 'AMT_REQ_CREDIT_BUREAU_DAY : Nombre de demandes de renseignements auprès du bureau de crédit dans la même journée',
    'AMT_REQ_CREDIT_BUREAU_WEEK': 'AMT_REQ_CREDIT_BUREAU_WEEK : Nombre de demandes de renseignements auprès du bureau de crédit dans la même semaine',
    'AMT_REQ_CREDIT_BUREAU_MON': 'AMT_REQ_CREDIT_BUREAU_MON : Nombre de demandes de renseignements auprès du bureau de crédit dans le même mois',
    'AMT_REQ_CREDIT_BUREAU_QRT': 'AMT_REQ_CREDIT_BUREAU_QRT : Nombre de demandes de renseignements auprès du bureau de crédit dans le même trimestre',
    'AMT_REQ_CREDIT_BUREAU_YEAR': 'AMT_REQ_CREDIT_BUREAU_YEAR : Nombre de demandes de renseignements auprès du bureau de crédit dans la même année',
}

# Sélectionnez un client avec Streamlit
st.title('Prédiction de Remboursement des Clients')
st.sidebar.title('Sélectionnez un client:')
client_ids = X_test['SK_ID_CURR'].tolist()  # Obtenez la liste des valeurs SK_ID_CURR

# Sélectionnez un client dans la liste déroulante
selected_client_id = st.sidebar.selectbox("Sélectionnez un client par SK_ID_CURR:", client_ids)

# Function to display capacity as a bar chart
def display_capacity_bar(proba):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(['Capacité de remboursement', 'Risque de défaut'], [proba, 1 - proba], color=['blue', 'r'])
    ax.set_title('Capacité de remboursement du client')
    ax.set_ylabel('Probabilité')
    ax.set_ylim(0, 1)

    for i, v in enumerate([proba, 1 - proba]):
        ax.text(i, v + 0.01, f'{v*100:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y*100:.0f}%'))

    st.pyplot(fig)

if selected_client_id:
    st.write(f"Informations pour le client sélectionné ({selected_client_id}):")

    # Obtenir une prédiction à partir de l'API Flask
    prediction_url = f'{api_url_predict}{selected_client_id}'
    prediction_response = requests.get(prediction_url)
    if prediction_response.status_code == 200:
        prediction = prediction_response.json()['prediction']
        st.write(f"Prédiction de défaut de paiement : {prediction}")
    else:
        st.error("Erreur lors de la récupération de la prédiction depuis l'API.")

    # Vérifiez si l'ID du client sélectionné existe dans le DataFrame X_test
    if selected_client_id in X_test['SK_ID_CURR'].values:
        # Obtenir la capacité de remboursement à partir du modèle
        client_data = X_test[X_test['SK_ID_CURR'] == selected_client_id]
        capacite = model.predict_proba(client_data)[:, 1]
        st.write(f"Capacité de remboursement : {capacite[0]:.2f}")
        display_capacity_bar(capacite[0])

        # Utilisez SHAP pour obtenir les valeurs SHAP
        explainer = shap.Explainer(model)
        shap_values = explainer.shap_values(client_data)

        # Créez un graphique SHAP
        plt.figure()
        shap.summary_plot(shap_values, client_data, plot_type='bar')

        # Affichez le graphique SHAP dans Streamlit
        st.pyplot(plt)

        # Utilisez SHAP pour obtenir les valeurs SHAP absolues moyennes
        shap_mean = np.abs(shap_values).mean(axis=0)

        # Triez les caractéristiques par importance décroissante
        feature_names = client_data.columns
        sorted_indices = np.argsort(shap_mean)[::-1]
        top_10_indices = sorted_indices[:10]
        top_10_features_names = [variable_names.get(feature, feature) for feature in feature_names[top_10_indices]]
        top_10_shap_mean = shap_mean[top_10_indices]

        # Créez un DataFrame pour afficher les 10 principales caractéristiques et leurs valeurs SHAP moyennes
        top_10_features_df = pd.DataFrame({'Variable : Définition': top_10_features_names, 'Valeur SHAP moyenne': top_10_shap_mean})
        st.write('Top 10 des variables les plus importantes :')
        st.table(top_10_features_df)

    else:
        st.error("ID du client sélectionné non trouvé dans les données de test.")

 # Fonction pour obtenir les clients similaires à partir de l'API
def get_similar_clients(SK_ID_CURR):
    url = f'http://localhost:5000/api/similar-clients/{SK_ID_CURR}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data['similar_clients']
    else:
        return None

# Demande à l'utilisateur de saisir SK_ID_CURR
SK_ID_CURR = st.number_input("Entrez SK_ID_CURR:", min_value=selected_client_id)

if st.button("Afficher quelques exemples de clients similaires"):
    similar_clients = get_similar_clients(selected_client_id)

    if similar_clients:
        # Créer une liste des 10 premiers SK_ID_CURR des clients similaires
        similar_client_ids = [client['SK_ID_CURR'] for client in similar_clients[:5]]
        
        # Exclure le client de base de la liste (s'il est présent)
        similar_client_ids = [client_id for client_id in similar_client_ids if client_id != selected_client_id]
        
        for client_id in similar_client_ids:
            st.write(client_id)
    else:
        st.write("Aucun client similaire trouvé.")



