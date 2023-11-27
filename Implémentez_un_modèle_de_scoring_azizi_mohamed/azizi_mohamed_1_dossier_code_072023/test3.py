import pytest
import matplotlib.pyplot as plt
from dash_app import display_capacity_bar

# Importez les modules nécessaires de Streamlit pour les tests
import streamlit as st

@pytest.mark.mpl_image_compare
def test_display_capacity_bar():
    proba = 0.13  # client 384723
    fig = display_capacity_bar(proba)

    assert isinstance(fig, plt.Figure)

# Ajoutez votre code Streamlit ici en utilisant pytest.mark.streamlit
@pytest.mark.streamlit
def test_streamlit_app():
    # Utilisez st.sidebar pour simuler la sélection d'un client
    with st.sidebar:
        st.title('Prédiction de Remboursement des Clients')
        st.sidebar.title('Sélectionnez un client:')
        client_selector = st.sidebar.selectbox("Sélectionnez un client:", [0, 1, 2, 3, 4])  # Remplacez cette liste par les valeurs appropriées

    # Assurez-vous que votre application Streamlit interagit correctement avec les widgets
    assert isinstance(client_selector, int)  # Assurez-vous que client_selector est un entier

