import streamlit as st
import pandas as pd  
from pycaret.clustering import setup, create_model, assign_model, plot_model, save_model, load_model, predict_model
import plotly.express as px
import json

# cd od_zera_do_ai/modul_7/FF2
# streamlit run --server.headless=true app.py

# ścieżki do danych i wytrenowanego modelu
MODEL_NAME = 'welcome_survey_clustering_pipeline_v2'
DATA = 'welcome_survey_simple_v2.csv'
CLUSTER_NAMES_AND_DESCRIPTIONS = 'welcome_survey_cluster_names_and_descriptions_v2.json'

# wczytywanie wytrenowanego modelu
@st.cache_data
def get_model():
    return load_model(MODEL_NAME)

# wczytywanie danych 
@st.cache_data
def get_all_participants():
    model = get_model()
    all_df = pd.read_csv(DATA, sep=';')
    df_with_clusters = predict_model(model, data=all_df)
    return df_with_clusters

# dodanie funkcji do przydzielania opisów klastrów 
@st.cache_data
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding='utf-8') as f:
        return json.loads(f.read())

# przypisanie oryginalnych calych danych do zmiennej 
all_df = get_all_participants()

# =================================
# sidebrar
# =================================
with st.sidebar:
    st.header("Powiedz nam coś o sobie")
    st.markdown("Pomożemy Ci znaleźć osoby, które mają podobne zainteresowania")
    age = st.selectbox("Wiek", sorted(all_df['age'].dropna().unique()))
    edu_level = st.selectbox("Wykształcenie", sorted(all_df['edu_level'].dropna().unique()))
    fav_animals = st.selectbox("Ulubione zwierzęta", sorted(all_df['fav_animals'].dropna().unique()))
    fav_place = st.selectbox("Ulubione miejsce", sorted(all_df['fav_place'].dropna().unique()))
    gender = st.radio("Płeć", sorted(all_df['gender'].dropna().unique()))

    person_df = pd.DataFrame([
        {
            'age': age,
            'edu_level': edu_level,
            'fav_animals': fav_animals,
            'fav_place': fav_place,
            'gender': gender
        }
    ])


# =================================
# main
# =================================

# wyswietla wybrane dane 
st.write("Wybrane dane:")
st.dataframe(person_df, hide_index=True)

# wywolanie funckji 
cluster_names_and_descriptions = get_cluster_names_and_descriptions()
model = get_model()
predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]
predicted_cluster_data = cluster_names_and_descriptions[predicted_cluster_id]

# odfiltrowanie danych do rekordow pasujacych do klastra 
st.header(f"Najbliżej jest Tobie do grupy: {predicted_cluster_data['name']}")
st.markdown(predicted_cluster_data['description'])
same_cluster_df = all_df[all_df['Cluster'] == predicted_cluster_id]
st.metric('Liczba Twoich znajomych', len(same_cluster_df))
st.dataframe(same_cluster_df, hide_index=True)


# wykresy harakteryzujace klaster 
st.header("Osoby z grupy")
fig = px.histogram(same_cluster_df.sort_values("age"), x="age")
fig.update_layout(
    title="Rozkład wieku w grupie",
    xaxis_title="Wiek",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="edu_level")
fig.update_layout(
    title="Rozkład wykształcenia w grupie",
    xaxis_title="Wykształcenie",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="fav_animals")
fig.update_layout(
    title="Rozkład ulubionych zwierząt w grupie",
    xaxis_title="Ulubione zwierzęta",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="fav_place")
fig.update_layout(
    title="Rozkład ulubionych miejsc w grupie",
    xaxis_title="Ulubione miejsce",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="gender")
fig.update_layout(
    title="Rozkład płci w grupie",
    xaxis_title="Płeć",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)