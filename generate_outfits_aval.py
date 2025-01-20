from PIL import Image, UnidentifiedImageError
import numpy as np
import pandas as pd
import warnings

# Desactivar todos los warnings
warnings.filterwarnings("ignore")

import random
import tensorflow as tf
from siameses import OutfitRecommenderSiameses
import requests
from io import BytesIO
from autoencoder import OutfitRecommenderAutoencoder
import os
import time
"""---------------------------------------------------------------------"""
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

# Reintento automático para solicitudes HTTP
def get_image_with_retries(url, max_retries=3, delay=5):
    retries = 0
    while retries < max_retries:
        try:
            response = requests.get(url, timeout=7)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        except requests.exceptions.RequestException as e:
            retries += 1
            #print(f"Error al descargar la imagen: {e}. Reintento {retries}/{max_retries}...")
            time.sleep(delay)
        except UnidentifiedImageError:
            print("Error: La URL no contiene una imagen válida.")
            return None
    print("Error: No se pudo descargar la imagen después de varios intentos.")
    return None



path_resources = os.getcwd() + "/resources/"

catalog_all=pd.read_csv(path_resources+"df_fashion_v3.csv")
num_samples=130
samples=catalog_all.sample(num_samples)

# DataFrames para guardar outfits
outfits_autoencoder = []
outfits_siameses = []
outfit_id = 0

columnas_letab=["gender", "subCategory", "articleType", "season", "usage", "Color"]

# Configuración de reintentos
max_retries = 3  # Número máximo de reintentos
retry_delay = 5  # Tiempo entre reintentos (en segundos)


autoencoder_model = OutfitRecommenderAutoencoder(path_resources)
siameses_model=OutfitRecommenderSiameses(path_resources)


for index, df_data_prenda in samples.iterrows():
    print(outfit_id)
    # Descargar imagen con reintentos
    image = get_image_with_retries(df_data_prenda["link"], max_retries=3, delay=3)
    if image is None:
        continue  # Saltar esta iteración si no se pudo descargar la imagen

    df_data_prenda = pd.DataFrame([df_data_prenda]).drop(["Unnamed: 0", "filename","image"], axis=1, errors="ignore")

    # Procesar datos con Autoencoder
    input_data = df_data_prenda.copy(deep=True)
    input_data[columnas_letab] = preprocess_input_data(df_data_prenda[columnas_letab], autoencoder_model)
    input_data_drop = input_data.drop(["image", "id", "link"], errors="ignore", axis=1)
    tensor_tuple = get_tensor_tuple(input_data_drop)
    input_embedding = autoencoder_model.get_embedding(image, tensor_tuple) #input_embeddings np.array (1,250)

    indices, scores = autoencoder_model.iterative_max_score_selection(tensor_tuple, input_embedding)
    data_indices = autoencoder_model.catalog_tab_all[indices]
    df_outfit_autoencoder = pd.DataFrame(data_indices, columns=["id", "gender", "subCategory", "articleType", "season", "usage", "Color"])

    # Agregar links e ID de outfit
    df_outfit_autoencoder["link"] = catalog_all.loc[catalog_all["id"].isin(df_outfit_autoencoder["id"]), "link"].values
    df_outfit_autoencoder = reverse_transform(df_outfit_autoencoder, autoencoder_model)
    df_outfit_autoencoder=pd.concat([df_data_prenda,df_outfit_autoencoder])
    df_outfit_autoencoder=df_outfit_autoencoder.reset_index()
    df_outfit_autoencoder["outfit_id"] = outfit_id
    df_outfit_autoencoder["embedding"] = None
    df_outfit_autoencoder.at[0,"embedding"]= list(np.array([np.float32(val) for val in input_embedding]).squeeze())


    outfits_autoencoder.append(df_outfit_autoencoder)

    # Procesar datos con Siameses
    input_data_drop = df_data_prenda.iloc[0].drop(["image", "link"], errors="ignore", axis=0)
    outfit, _, _ = siameses_model.recommend_outfit(input_data_drop, image)
    df_outfit_siameses = pd.DataFrame(outfit)

    # Agregar links e ID de outfit
    df_outfit_siameses["link"] = catalog_all.loc[catalog_all["id"].isin(df_outfit_siameses["id"]), "link"].values
    df_outfit_siameses["outfit_id"] = outfit_id
    df_outfit_siameses=df_outfit_siameses.reset_index()
    df_outfit_siameses["embedding"] = None
    df_outfit_siameses.at[0,"embedding"]= list(np.array([np.float32(val) for val in siameses_model.get_embedding(input_data_drop, image)]).squeeze())

    outfits_siameses.append(df_outfit_siameses)
    outfit_id += 1



# Consolidar DataFrames finales
outfits_autoencoder_df = pd.concat(outfits_autoencoder, ignore_index=True)
outfits_siameses_df = pd.concat(outfits_siameses, ignore_index=True)


outfits_autoencoder_df = outfits_autoencoder_df[outfits_autoencoder_df['outfit_id'] < 100]
outfits_siameses_df = outfits_siameses_df[outfits_siameses_df['outfit_id'] < 100]


""".------------------------------------------------"""

catalogo_siameses=siameses_model.catalogo.copy(deep=True)[["id", "embedding"]]
catalogo_siameses['embedding']=catalogo_siameses['embedding'].apply(list)

missing_embeddings = outfits_siameses_df['embedding'].isnull()

# Crear un DataFrame temporal con los valores del catálogo que coinciden con los ids faltantes
missing_ids = outfits_siameses_df.loc[missing_embeddings, 'id']
fill_values = catalogo_siameses[catalogo_siameses['id'].isin(missing_ids)]

# Rellenar los valores faltantes utilizando el catálogo
outfits_siameses_df = outfits_siameses_df.set_index('id')
fill_values = fill_values.set_index('id')
outfits_siameses_df.update(fill_values)
outfits_siameses_df = outfits_siameses_df.reset_index()


""".------------------------------------------------"""

catalog_autoencoder=autoencoder_model.data_all.copy(deep=True)[["id", "embedding"]]


catalog_autoencoder["embedding"]=catalog_autoencoder["embedding"].apply(list)

missing_embeddings = outfits_autoencoder_df['embedding'].isnull()

# Crear un DataFrame temporal con los valores del catálogo que coinciden con los ids faltantes
missing_ids = outfits_autoencoder_df.loc[missing_embeddings, 'id']
fill_values = catalog_autoencoder[catalog_autoencoder['id'].isin(missing_ids)]

# Rellenar los valores faltantes utilizando el catálogo
outfits_autoencoder_df = outfits_autoencoder_df.set_index('id')
fill_values = fill_values.set_index('id')
outfits_autoencoder_df.update(fill_values)
outfits_autoencoder_df = outfits_autoencoder_df.reset_index()

outfits_autoencoder_df.drop(["index"],axis=1,inplace=True)
outfits_siameses_df.drop(["index"],axis=1,inplace=True)

outfits_autoencoder_df.to_csv(path_resources + "outfits_generate_autoencoder.csv", index=False)
outfits_siameses_df.to_csv(path_resources + "outfits_generate_siameses.csv", index=False)



# Filtrar categorías
Topwear_cat = catalog_all[catalog_all["subCategory"] == "Topwear"]
Bottomwear_cat = catalog_all[catalog_all["subCategory"] == "Bottomwear"]
Shoes_cat = catalog_all[catalog_all["subCategory"] == "Shoes"]
over_Topwear_cat = catalog_all[catalog_all["subCategory"] == "over_Topwear"]
Dress_cat = catalog_all[catalog_all["subCategory"] == "Dress"]

# Lista para guardar los outfits
outfits = []
outfit_id = 100

# Generar varios outfits
num_outfits_ran = 100  # Cambiar según el número de outfits deseados
for _ in range(num_outfits_ran):
    # Seleccionar una fila aleatoria de cada categoría necesaria
    outfit = catalog_all.sample(4).assign(outfit_id=outfit_id)
 
    outfits.append(outfit)
    outfit_id += 1

# Combinar todos los outfits en un solo DataFrame
outfits_aleatorios_df = pd.concat(outfits, ignore_index=True)
outfits_aleatorios_df.to_csv(path_resources + "outfits_generate_random.csv", index=False)

