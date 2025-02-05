import time
import streamlit as st
from PIL import Image
import pandas as pd
import json

import tensorflow as tf
from siameses import OutfitRecommenderSiameses
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from autoencoder import OutfitRecommenderAutoencoder
import os


# Claus esperades al JSON
EXPECTED_KEYS = {"id", "gender", "subCategory", "articleType", "season", "usage", "Color" }

# Títol de l'aplicació
st.markdown("<h1 style='text-decoration: underline;'>OUTFITMATCH</h1>", unsafe_allow_html=True)

st.markdown("### Carrega una imatge i un fitxer JSON")
uploaded_image = st.file_uploader("Puja una imatge (formats: jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Mostrar la imatge carregada
    image = Image.open(uploaded_image)
    st.image(image, caption="Imatge carregada", width=300)

uploaded_json = st.file_uploader("Puja un fitxer JSON amb dades tabulars", type=["json"])

if uploaded_json is not None:
    try:
        # Llegir el fitxer JSON
        data_prenda = json.load(uploaded_json)

        # Verificar que el JSON és un diccionari (una sola instància)
        if not isinstance(data_prenda, dict):
            st.error("El JSON ha de contenir una sola instància representada com un diccionari.")
        else:
            # Verificar que conté totes les claus esperades
            keys = set(data_prenda.keys())
            missing_keys = EXPECTED_KEYS - keys
            if missing_keys:
                st.error(f"El JSON no té les claus esperades. Falta: {', '.join(missing_keys)}")
            else:
                # Convertir les dades a un DataFrame per a visualització
                df_data_prenda = pd.DataFrame([data_prenda])  # Convertir a una sola fila en DataFrame
                # Mostrar el DataFrame
                st.write("Contingut del JSON:")
                st.dataframe(df_data_prenda)
    except Exception as e:
        st.error(f"Error en processar el fitxer JSON: {e}")


# Defineix funcions d'ajuda
def preprocess_input_data(input_data, autoencoder_model):
    input_data = input_data.copy(deep=True).map(lambda x: x.lower() if isinstance(x, str) else x)
    for col in autoencoder_model.le_tab:
        input_data[col] = autoencoder_model.le_tab[col].transform(input_data[col])
    return input_data

def get_tensor_tuple(input_data):
    return tuple(tf.convert_to_tensor(value, dtype=tf.int32) for value in input_data.values.flatten())

def reverse_transform(df, autoencoder_model):
    for col in autoencoder_model.le_tab:
        df[col] = autoencoder_model.le_tab[col].inverse_transform(df[col])
    return df

def merge_with_links(df_outfit, path_resources):
    df_link = pd.read_csv(f"{path_resources}url_catalog.csv")
    df_link['id'] = df_link['filename'].str.replace('.jpg', '', regex=False)
    df_outfit['id'] = df_outfit['id'].astype(str)
    df_link['id'] = df_link['id'].astype(str)
    return pd.merge(df_outfit, df_link, on='id', how='inner')


# Funció per descarregar la imatge amb retries utilitzant només requests
def fetch_image_with_retries(url, retries=3, delay=2):
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Verifica errors HTTP
            return Image.open(BytesIO(response.content))
        except requests.RequestException as e:
            if attempt < retries - 1:  # Si no és l'últim intent, espera i reintenta
                time.sleep(delay)
            else:  # Si és l'últim intent, llança l'excepció
                raise e

def display_images(df, url_column='link', type_columns='subCategory', max_images=4, img_width=200):
    cols_per_row = 2
    for i in range(0, min(len(df), max_images), cols_per_row):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            if i + j >= len(df):
                break
            try:
                img = fetch_image_with_retries(df.iloc[i + j][url_column])
                tipo_cand= df.iloc[i + j][type_columns]
                if i==0 and j==0:
                    cols[j].image(img, caption=f"Imatge d'entrada: {tipo_cand}", width=img_width)
                else:
                    cols[j].image(img, caption=f"Candidata {i + j}: {tipo_cand}", width=img_width)

            except Exception as e:
                cols[j].warning(f"No s'ha pogut carregar la imatge {i + j + 1}. Error: {e}")


st.markdown("### Selecciona el model per generar l'outfit")
model_option = st.selectbox("Tria un model", ["Autoencoder", "Siameses"])  # Canviat a selectbox

if st.button("Generar Outfit"):
    if uploaded_image is not None and uploaded_json is not None:
        path_resources = os.getcwd() + "/resources/"

        if model_option == "Autoencoder":
            st.markdown("##### Generant outfit amb el model Autoencoder...")

            autoencoder_model = OutfitRecommenderAutoencoder(path_resources) # df_data_prenda, image

            input_data = preprocess_input_data(df_data_prenda, autoencoder_model)
            input_data_drop = input_data.drop(["image", "id"],errors="ignore", axis=1)

            tensor_tuple = get_tensor_tuple(input_data_drop)
            input_embedding = autoencoder_model.get_embedding(image, tensor_tuple)
            indices, scores = autoencoder_model.iterative_max_score_selection(tensor_tuple, input_embedding)

            data_indices = autoencoder_model.catalog_tab_all[indices]
            df_outfit = pd.DataFrame(data_indices, columns=["id", "gender", "subCategory", "articleType", "season", "usage", "Color"])

            input_data = input_data.drop(["image"],errors="ignore", axis=1)
            df_outfit = pd.concat([input_data, df_outfit])
            df_outfit = reverse_transform(df_outfit, autoencoder_model)
            df_merged = merge_with_links(df_outfit, path_resources)
            st.write(df_merged)

            st.markdown("### Visualització d'Outfit - Autoencoder")
            display_images(df_merged)

        elif model_option == "Siameses":
            st.markdown("##### Generant outfit amb el model Siameses...")

            siameses_model = OutfitRecommenderSiameses(path_resources)
            outfit, _ , cos= siameses_model.recommend_outfit(df_data_prenda.iloc[0].drop(["image"],errors="ignore", axis=0), image)
            df_outfit = pd.DataFrame(outfit)
            df_outfit = merge_with_links(df_outfit, path_resources)
            st.write(df_outfit)

            st.markdown("### Visualització d'Outfit - Siameses")
            st.write("Hay imagenes que no concuerdan con el tipo, es un error del dataset")

            display_images(df_outfit)
