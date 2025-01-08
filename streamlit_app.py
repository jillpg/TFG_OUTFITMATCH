import streamlit as st
from PIL import Image
import pandas as pd
import json
from siameses import OutfitRecommenderSiameses

# Claves esperadas en el JSON
EXPECTED_KEYS = {"id", "gender", "subCategory", "articleType", "season", "usage", "Color"}

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
        data = json.load(uploaded_json)

        # Verificar que el JSON es un diccionario (una sola instancia)
        if not isinstance(data, dict):
            st.error("El JSON debe contener una sola instancia representada como un diccionario.")
        else:
            # Verificar que contiene todas las claves esperadas
            keys = set(data.keys())
            missing_keys = EXPECTED_KEYS - keys
            if missing_keys:
                st.error(f"El JSON no tiene las claves esperadas. Faltan: {', '.join(missing_keys)}")
            else:
                # Convertir los datos a un DataFrame para visualización
                df = pd.DataFrame([data])  # Convertir a una sola fila en DataFrame
                # Mostrar el DataFrame
                st.write("Contenido del JSON:")
                st.dataframe(df)
    except Exception as e:
        st.error(f"Error al procesar el archivo JSON: {e}")

siameses_model= OutfitRecommenderSiameses("/workspaces/TFG_OUTFITMATCH/resources/")



