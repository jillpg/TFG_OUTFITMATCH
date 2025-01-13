import streamlit as st
from PIL import Image
import pandas as pd
import json
from siameses import OutfitRecommenderSiameses
import matplotlib.pyplot as plt
import requests
from io import BytesIO


# Claves esperadas en el JSON
EXPECTED_KEYS = {"id", "gender", "subCategory", "articleType", "season", "usage", "Color" }

# Título de la aplicación
st.title("Cargar una imagen y un archivo JSON")

# Sección para cargar una imagen
st.header("Cargar Imagen")
uploaded_image = st.file_uploader("Sube una imagen (formatos: jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Mostrar la imagen cargada
    image = Image.open(uploaded_image)
    st.image(image, caption="Imagen cargada", use_container_width=True)

# Separador
st.markdown("---")

# Sección para cargar un archivo JSON
st.header("Cargar archivo JSON")
uploaded_json = st.file_uploader("Sube un archivo JSON con datos tabulares", type=["json"])

if uploaded_json is not None:
    try:
        # Leer el archivo JSON
        data_prenda = json.load(uploaded_json)

        # Verificar que el JSON es un diccionario (una sola instancia)
        if not isinstance(data_prenda, dict):
            st.error("El JSON debe contener una sola instancia representada como un diccionario.")
        else:
            # Verificar que contiene todas las claves esperadas
            keys = set(data_prenda.keys())
            missing_keys = EXPECTED_KEYS - keys
            if missing_keys:
                st.error(f"El JSON no tiene las claves esperadas. Faltan: {', '.join(missing_keys)}")
            else:
                # Convertir los datos a un DataFrame para visualización
                df_data_prenda = pd.DataFrame([data_prenda])  # Convertir a una sola fila en DataFrame
                # Mostrar el DataFrame
                st.write("Contenido del JSON:")
                st.dataframe(df_data_prenda)
    except Exception as e:
        st.error(f"Error al procesar el archivo JSON: {e}")


if uploaded_image is not None and uploaded_json is not None:
    path_resources="/workspaces/TFG_OUTFITMATCH/resources/"
    siameses_model= OutfitRecommenderSiameses(path_resources)
    outfit, _ = siameses_model.recommend_outfit(df_data_prenda.iloc[0].drop(["image"], axis=0),image)
    df_outfit=pd.DataFrame(outfit)
    

    df_link=pd.read_csv(path_resources+"url_catalog.csv")
    df_link['id'] = df_link['filename'].str.replace('.jpg', '', regex=False)
    df_outfit['id'] = df_outfit['id'].astype(str)
    df_link['id'] = df_link['id'].astype(str)
    df_merged = pd.merge(df_outfit, df_link, on='id', how='inner')
    st.dataframe(df_merged)
    # Función para mostrar imágenes desde URLs
    def mostrar_imagenes(df, url_column='link', max_images=4):
        for i in range(min(len(df), max_images)):
            try:
                response = requests.get(df.iloc[i][url_column])
                img = Image.open(BytesIO(response.content))
                st.image(img, caption=f"Imagen {i+1}", use_container_width=True)
            except Exception as e:
                st.warning(f"No se pudo cargar la imagen {i+1}. Error: {e}")

    # Llamar a la función para mostrar imágenes
    st.title("Visualización de Outfit")
    mostrar_imagenes(df_merged)


#import tensorflow as tf
#path="/workspaces/TFG_OUTFITMATCH/resources/encoder_combined_autoencoder.keras"
#modelo_prueba=tf.keras.models.load_model(path)
