import pandas as pd
import tensorflow as tf
import numpy as np
import pickle
class OutfitRecommenderAutoencoder:
    def __init__(self, resources_dir):
        # Rutas de los recursos
        encoder_model_path = resources_dir + "autoencoder_encoder_combined.keras"
        embeddings_path = resources_dir + "autoencoder_catalog_emb.npy"
        tabular_data_path_all = resources_dir + "autoencoder_catalog_tab.npy"
        encoders_path = resources_dir + "autoencoder_encoders.pkl"
        compatibility_tensors_path = resources_dir + "autoencoder_compatibility.pkl"

        # Cargar modelos y datos
        self.encoder_combined = tf.keras.models.load_model(encoder_model_path)
        self.catalog_emb = np.load(embeddings_path)
        self.catalog_tab_all = np.load(tabular_data_path_all)
        self.catalog_tab = self.catalog_tab_all[1:]
        self.catalog_tab_all = self.catalog_tab_all.T
        self.target_size=(224,224)
        self.model_columns=["id", "gender", "subCategory", "articleType", "season", "usage", "Color"]
        self.data_all=pd.DataFrame(self.catalog_tab_all, columns=self.model_columns)
        self.data_all["embedding"]=list(self.catalog_emb)


        with open(encoders_path, "rb") as file:
            self.le_tab = pickle.load(file)
        with open(compatibility_tensors_path, "rb") as file:
            self.compatibility_tensors = pickle.load(file)

        self.cat_cols = ['gender', 'subCategory', 'articleType', 'season', 'usage', 'Color']

    def cosine_similarity_matrix(self, input_embedding, catalog_embeddings):
        input_embedding_norm = tf.nn.l2_normalize(input_embedding, axis=-1)
        catalog_embeddings_norm = tf.nn.l2_normalize(catalog_embeddings, axis=-1)
        return tf.matmul(input_embedding_norm, catalog_embeddings_norm, transpose_b=True)

    def calculate_compatibility_with_embeddings(self, input_data, input_embedding):
        latent_similarity = self.cosine_similarity_matrix(input_embedding, self.catalog_emb)
        latent_similarity = tf.squeeze(latent_similarity)

        # Inicializar penalizaciones
        usage_catalog = self.catalog_tab[self.cat_cols.index('usage')]
        gender_catalog = self.catalog_tab[self.cat_cols.index('gender')]
        article_catalog = self.catalog_tab[self.cat_cols.index('articleType')]
        color_catalog = self.catalog_tab[self.cat_cols.index('Color')]
        season_catalog = self.catalog_tab[self.cat_cols.index('season')]

        # Índices de metadatos de la prenda de entrada
        usage_input = tf.broadcast_to(input_data[self.cat_cols.index('usage')], tf.shape(usage_catalog))
        gender_input = tf.broadcast_to(input_data[self.cat_cols.index('gender')], tf.shape(gender_catalog))
        article_input = tf.broadcast_to(input_data[self.cat_cols.index('articleType')], tf.shape(article_catalog))
        color_input = tf.broadcast_to(input_data[self.cat_cols.index('Color')], tf.shape(color_catalog))
        season_input = tf.broadcast_to(input_data[self.cat_cols.index('season')], tf.shape(season_catalog))

        # Penalizaciones vectorizadas
        usage_simil = tf.gather_nd(
            self.compatibility_tensors['usage1_usage2'], tf.stack([usage_input, usage_catalog], axis=1)
        )
        gender_simil = tf.gather_nd(
            self.compatibility_tensors['gender1_gender2'], tf.stack([gender_input, gender_catalog], axis=1)
        )
        article_simil = tf.gather_nd(
            self.compatibility_tensors['articleType1_articleType2'], tf.stack([article_input, article_catalog], axis=1)
        )
        article_usage_simil = tf.gather_nd(
            self.compatibility_tensors['articleType1_usage2'],
            tf.stack([article_input, usage_catalog], axis=1)
        )
        article_usage_usage_simil = tf.gather_nd(
            self.compatibility_tensors['articleType1_usage1_usage2'],
            tf.stack([article_input, usage_input, usage_catalog], axis=1)
        )
        article_usage_article_simil = tf.gather_nd(
            self.compatibility_tensors['articleType1_usage1_articleType2'],
            tf.stack([article_input, usage_input, article_catalog], axis=1)
        )
        color_simil = tf.gather_nd(
            self.compatibility_tensors['Color1_Color2'],
            tf.stack([color_catalog, color_input], axis=1)
        )
        season_simil = tf.gather_nd(
            self.compatibility_tensors['season1_season2'],
            tf.stack([season_catalog, season_input], axis=1)
        )

        total_simil = (usage_simil +
                       gender_simil +
                       article_simil +
                       article_usage_simil +
                       article_usage_usage_simil +
                       article_usage_article_simil +
                       color_simil + season_simil) / len(self.compatibility_tensors)

        compatibility_scores = (latent_similarity * 0.3) + (total_simil * 0.7)
        return compatibility_scores, latent_similarity, total_simil

    def iterative_max_score_selection(self, input_data, input_embedding):
        selected_indices = []
        selected_subcategories = set()
        selected_subcategories.add(input_data[self.cat_cols.index('subCategory')].numpy())

        accumulated_score = 0.0
        clases_sub = self.le_tab['subCategory'].classes_
        d_idx = np.where(clases_sub == 'dress')[0][0]
        t_idx = np.where(clases_sub == 'topwear')[0][0]
        b_idx = np.where(clases_sub == 'bottomwear')[0][0]

        subcategory_restrictions = {
            d_idx: {t_idx, b_idx},
            t_idx: {d_idx},
            b_idx: {d_idx}
        }

        subcategories = self.catalog_tab[self.cat_cols.index('subCategory')]

        while True:
            scores, _, _ = self.calculate_compatibility_with_embeddings(input_data, input_embedding)

            mask = ~tf.reduce_any(tf.equal(tf.range(tf.shape(scores)[0])[:, None], selected_indices), axis=1)
            valid_subcategory_mask = [
                subcategories[i] not in selected_subcategories and
                all(subcategories[i] not in subcategory_restrictions.get(sub, set())
                    for sub in selected_subcategories)
                for i in range(len(subcategories))
            ]

            combined_mask = tf.constant(mask) & tf.constant(valid_subcategory_mask, dtype=tf.bool)

            if not tf.reduce_any(combined_mask):
                #print("No hay más elementos válidos para seleccionar. Finalizando...")
                break

            masked_scores = tf.where(combined_mask, scores, tf.zeros_like(scores))
            max_score_idx = tf.argmax(masked_scores).numpy()
            max_score_value = masked_scores[max_score_idx].numpy()

            if len(selected_indices) > 0:
                accumulated_score =  (accumulated_score + max_score_value) / 2
            else:
                accumulated_score=max_score_value

            input_embedding = self.catalog_emb[max_score_idx:max_score_idx + 1]
            input_data = [x[max_score_idx] for x in self.catalog_tab]

            selected_indices.append(max_score_idx)
            selected_subcategories.add(subcategories[max_score_idx])

     
        return selected_indices, accumulated_score

    def get_embedding(self, input_image, input_data):
        image = tf.convert_to_tensor(input_image, dtype=tf.uint8)
        image = tf.image.resize(image, self.target_size)
        image = tf.keras.applications.resnet50.preprocess_input(image)  # Preprocesar para ResNet50
        inputs = [tf.expand_dims(image, axis=0)] + [tf.expand_dims(data, axis=0) for data in input_data]
        return self.encoder_combined.predict(inputs)
