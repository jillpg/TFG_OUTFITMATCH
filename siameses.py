import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import os
class OutfitRecommenderSiameses:
    def __init__(self, path_resources):
        print(os.getcwd())  # Obtiene el directorio actual
        self.path_resources = path_resources
        self.catalogo_path = path_resources + "catalogo_siameses.csv"
        self.model_path = path_resources + "model_siameses.keras"
        self.encoders_path = path_resources + "siameses_encoder.pkl"
        self.catalogo = self.load_catalogo()
        self.encoders = self.load_encoders()
        self.base_model = self.load_model()

    def load_catalogo(self):
        return pd.read_csv(self.catalogo_path)

    def load_encoders(self):
        with open(self.encoders_path, 'rb') as file:
            return pickle.load(file)

    def load_model(self):
        return tf.keras.models.load_model(self.model_path)

    def preprocess_input_resnet(self, img):
        """Normalización de imagen para ResNet50."""
        return tf.keras.applications.resnet50.preprocess_input(img)

    def load_and_preprocess_image(self, img):
        """Carga y preprocesa una imagen."""
        img = img.resize((224, 224))  # Para ResNet50
        img = np.array(img, dtype=np.float32)
        img = self.preprocess_input_resnet(img)
        return img

    def get_embedding(self, prenda, image):
        """
        Obtiene el embedding de una prenda utilizando un modelo base.
        """
        prenda_2 = prenda.copy()
        img = self.load_and_preprocess_image(image)
        img = tf.expand_dims(img, axis=0)  # Expande dimensión para el batch

        for col in self.encoders:
            prenda_2[col] = self.encoders[col].transform([prenda_2[col]])  # Transformación de atributos categóricos
        attrs = np.expand_dims(
            np.array(prenda_2.drop(["id", "image"], axis=0), dtype=np.int32), axis=0
        )
        emb = self.base_model.predict([img, attrs], verbose=0)
        return emb

    def find_most_compatible(self, emb_ref):
        """Encuentra la prenda más compatible con base en el embedding de referencia."""
        embeddings_matrix = np.vstack(self.catalogo["embedding"])
        cosine_similarities = np.dot(embeddings_matrix, emb_ref.T)
        idx_max_similarity = np.argmax(cosine_similarities)
        return self.catalogo.iloc[idx_max_similarity]

    def average_embeddings(self, embeddings):
        """Calcula el promedio de una lista de embeddings."""
        return np.mean(np.stack(embeddings, axis=0), axis=0)

    def filtrar_por_categoria(self, categorias):
        """Filtra el catálogo eliminando prendas que coincidan con las categorías dadas."""
        return self.catalogo[~self.catalogo['subCategory'].isin(categorias)]

    def recommend_outfit(self, prenda_inicial,image_inicial):
        """
        Genera una recomendación de outfit basada en una prenda inicial.
        """
        # Inicializa el outfit y el embedding
        emb_outfit = self.get_embedding(prenda_inicial, image_inicial).squeeze()
        filter_categories = [prenda_inicial.subCategory]
        outfit = [prenda_inicial]

        while True:
            # Lógica de categorías
            if ("Bottomwear" in filter_categories or "Topwear" in filter_categories) and "Dress" not in filter_categories:
                filter_categories.append("Dress")
            elif "Dress" in filter_categories and ("Topwear" not in filter_categories and  "Bottomwear" not in filter_categories):
                filter_categories.append("Bottomwear")
                filter_categories.append("Topwear")

            # Filtra el catálogo
            filtered_catalogo = self.filtrar_por_categoria(filter_categories)

            # Si el catálogo está vacío, termina el bucle
            if filtered_catalogo.empty:
                break

            # Encuentra la prenda más compatible
            selected = self.find_most_compatible(emb_outfit)

            # Actualiza el embedding del outfit
            emb_outfit = self.average_embeddings([emb_outfit, selected["embedding"]])

            # Actualiza las categorías y agrega la prenda al outfit
            filter_categories.append(selected.subCategory)
            outfit.append(selected)

            # Limita el tamaño del outfit
            if len(outfit) >= 4:
                break

        return outfit, filter_categories