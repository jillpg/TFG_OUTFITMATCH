import streamlit as st
from PIL import Image
import pandas as pd
import pickle

import tensorflow as tf
from siameses import OutfitRecommenderSiameses
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from autoencoder import OutfitRecommenderAutoencoder
import os
import requests
from io import BytesIO
import time

def validate_outfit_multidimensional(outfit, compatibility_matrices, encoders):
    """
    Calcula la compatibilidad promedio de un outfit usando matrices con campos diferenciados por _1 y _2.
    Convierte matrices tensoriales a numpy arrays antes de realizar cálculos.
    Usa LabelEncoder de sklearn para codificar los valores de los campos.
    """
    total_score = 0
    count = 0

    for i in range(len(outfit)):
        for j in range(i + 1, len(outfit)):
            prenda1 = outfit[i]
            prenda2 = outfit[j]

            # Validar compatibilidad en matrices de hasta 3 dimensiones
            for fields, matrix in compatibility_matrices.items():
                # Asegurar que la matriz sea un numpy array
                if not isinstance(matrix, np.ndarray):
                    matrix = matrix.numpy()

                # Extraer valores codificados de las prendas según el sufijo
                values = []
                for field in fields.split("_"):  # Separar los campos por guión bajo
                    if field.endswith('1'):
                        field_base = field[:-1]  # Quitar el sufijo '1'
                        encoded_value = encoders[field_base].transform([prenda1[field_base]])[0]
                        values.append(encoded_value)
                    elif field.endswith('2'):
                        field_base = field[:-1]  # Quitar el sufijo '2'
                        encoded_value = encoders[field_base].transform([prenda2[field_base]])[0]
                        values.append(encoded_value)
                    else:
                        raise ValueError(f"Campo {field} no tiene el formato esperado (_1 o _2).")

                # Calcular compatibilidad según la dimensión de la matriz
                if len(values) == 2:  # Matriz 2D
                    compatibility = matrix[values[0], values[1]]
                elif len(values) == 3:  # Matriz 3D
                    compatibility = matrix[values[0], values[1], values[2]]
                else:
                    raise ValueError("Solo se soportan matrices de 2 o 3 dimensiones.")

                total_score += compatibility
                count += 1

    return total_score / count if count > 0 else 0


def evaluate_recommendations_multidimensional(recommendations_df, compatibility_matrices, encoders, threshold=0.5):
    compliant = 0
    
    # Convertir el DataFrame en una lista de outfits
    grouped_recommendations = recommendations_df.groupby('outfit_id')  # Agrupa por una columna "outfit_id"
    for _, group in grouped_recommendations:
        outfit = group.to_dict(orient='records')  # Convierte el grupo a una lista de diccionarios
        score = validate_outfit_multidimensional(outfit, compatibility_matrices, encoders)
        if score >= threshold:  # Cumple si el score >= threshold
            compliant += 1
    
    compliance_rate = compliant / len(grouped_recommendations)
    return compliance_rate


# Define helper functions
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


path_resources = os.getcwd() + "/resources/"
# Cargar un archivo .pkl
with open(path_resources+'autoencoder_compatibility.pkl', 'rb') as file:
    compatibility_matrices = pickle.load(file)

with open(path_resources+'autoencoder_encoders.pkl', 'rb') as file:
    encoders = pickle.load(file)

catalog_all=pd.read_csv(path_resources+"df_fashion_v3.csv")
samples=catalog_all.sample(10)

# DataFrames para guardar outfits
outfits_autoencoder = []
outfits_siameses = []
outfit_id = 0

columnas_letab=["gender", "subCategory", "articleType", "season", "usage", "Color"]

autoencoder_model = OutfitRecommenderAutoencoder(path_resources)
siameses_model=OutfitRecommenderSiameses(path_resources)
for index, df_data_prenda in samples.iterrows():
    print(outfit_id)
    # Descargar imagen
    image_response = requests.get(df_data_prenda["link"], timeout=20)
    image_response.raise_for_status()  # Verifica errores HTTP
    image = Image.open(BytesIO(image_response.content))
    df_data_prenda=pd.DataFrame([df_data_prenda]).drop(["Unnamed: 0", "filename"],axis=1,errors="ignore")
    # Procesar datos con Autoencoder
    input_data=df_data_prenda.copy(deep=True)
    input_data[columnas_letab] = preprocess_input_data(df_data_prenda[columnas_letab], autoencoder_model)
    input_data_drop = input_data.drop(["image", "id", "link"], errors="ignore", axis=1)
    tensor_tuple = get_tensor_tuple(input_data_drop)
    input_embedding = autoencoder_model.get_embedding(image, tensor_tuple)

    indices, scores = autoencoder_model.iterative_max_score_selection(tensor_tuple, input_embedding)
    data_indices = autoencoder_model.catalog_tab_all[indices]
    df_outfit_autoencoder = pd.DataFrame(data_indices, columns=["id", "gender", "subCategory", "articleType", "season", "usage", "Color"])

    # Agregar links e ID de outfit
    df_outfit_autoencoder["link"] = catalog_all.loc[catalog_all["id"].isin(df_outfit_autoencoder["id"]), "link"].values
    df_outfit_autoencoder = reverse_transform(df_outfit_autoencoder, autoencoder_model)
    df_outfit_autoencoder["id_outfit"] = outfit_id
    outfits_autoencoder.append(df_outfit_autoencoder)

    # Procesar datos con Siameses
    input_data_drop = df_data_prenda.iloc[0].drop(["image", "link"], errors="ignore", axis=0)
    outfit, _ = siameses_model.recommend_outfit(input_data_drop, image)
    df_outfit_siameses = pd.DataFrame(outfit)

    # Agregar links e ID de outfit
    df_outfit_siameses["link"] = catalog_all.loc[catalog_all["id"].isin(df_outfit_siameses["id"]), "link"].values
    df_outfit_siameses["id_outfit"] = outfit_id
    outfits_siameses.append(df_outfit_siameses)
    outfit_id += 1
    time.sleep(5)


# Consolidar DataFrames finales
outfits_autoencoder_df = pd.concat(outfits_autoencoder, ignore_index=True)
outfits_siameses_df = pd.concat(outfits_siameses, ignore_index=True)

st.dataframe(outfits_autoencoder_df)
st.dataframe(outfits_siameses_df)


# Evaluar las recomendaciones
#compliance_rate = evaluate_recommendations_multidimensional(outfits_autoencoder_df, compatibility_matrices, encoders)
#print(f"Taxa de compliment: {compliance_rate:.2f}")
