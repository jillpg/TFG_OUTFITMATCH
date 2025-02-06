import pickle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import os

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
                        encoded_value = encoders[field_base].transform([prenda1[field_base].lower()])[0]
                        values.append(encoded_value)
                    elif field.endswith('2'):
                        field_base = field[:-1]  # Quitar el sufijo '2'
                        encoded_value = encoders[field_base].transform([prenda2[field_base].lower()])[0]
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

                total_score += (compatibility**0.6)
                count += 1

    return (total_score / count if count > 0 else 0)**(1/0.6)


def evaluate_recommendations_multidimensional(recommendations_df, compatibility_matrices, encoders, threshold=0.55):
    compliant = 0
    mean=[]
    # Convertir el DataFrame en una lista de outfits
    grouped_recommendations = recommendations_df.groupby('outfit_id')  # Agrupa por una columna "outfit_id"
    for _, group in grouped_recommendations:
        outfit = group.to_dict(orient='records')  # Convierte el grupo a una lista de diccionarios
        score = validate_outfit_multidimensional(outfit, compatibility_matrices, encoders)
        if score >= threshold:  # Cumple si el score >= threshold
            compliant += 1
        mean.append(score)
    compliance_rate = compliant / len(grouped_recommendations)
#    mean= mean  / len(grouped_recommendations)
    return compliance_rate , mean


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

outfits_autoencoder_df=pd.read_csv(path_resources+"outfits_generate_autoencoder.csv")
outfits_siameses_df=pd.read_csv(path_resources+"outfits_generate_siameses.csv")
outfits_aleatorios_df=pd.read_csv(path_resources+"outfits_generate_random.csv")
outfits_random_perClass_df=pd.read_csv(path_resources+"outfits_random_perClass.csv")


threshold=0.6
# Evaluar las recomendaciones
compliance_rate_autoeencoder, mean_autoeencoder= evaluate_recommendations_multidimensional(outfits_autoencoder_df, 
                                                                         compatibility_matrices, encoders,threshold)

with open(path_resources+'autoencoder_encoders.pkl', 'rb') as file:
    encoders = pickle.load(file)

# Evaluar las recomendaciones
compliance_rate_siameses, mean_siameses = evaluate_recommendations_multidimensional(outfits_siameses_df, 
                                                                     compatibility_matrices, encoders,threshold)


# Evaluar las recomendaciones
compliance_rate_random, mean_random = evaluate_recommendations_multidimensional(outfits_aleatorios_df, 
                                                                   compatibility_matrices, encoders,threshold)


compliance_rate_randomPerclass, mean_randomPerclass = evaluate_recommendations_multidimensional(outfits_random_perClass_df, 
                                                                   compatibility_matrices, encoders,threshold)


models = ['Autoencoder', 'Siameses' ,'Random']
accuracies = [compliance_rate_autoeencoder, compliance_rate_siameses ,compliance_rate_random]

# Crear el gráfico de barras
plt.figure(figsize=(8, 5))
plt.bar(models, accuracies, width=0.4)
plt.ylim(0, 1)  # Rango de 0 a 1 para representar accuracies
plt.title(f'Comparació de Consitència amb les Regles entre Models (Llindar: {threshold})')
plt.ylabel('Percentatge que superen el llindar')
plt.xlabel('Models')
plt.grid(axis='y', linestyle='--', alpha=0.7)
# Mostrar los valores sobre las barras
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=10)


plt.savefig(os.getcwd() + "/archivos_informe/grafico_reglas.jpg")



means = [mean_autoeencoder, mean_siameses ,mean_random]

# Crear el boxplot
plt.figure(figsize=(8, 5))
plt.boxplot(means, labels=models)

# Configuración del gráfico
plt.ylim(0, 1)  # Ajustar rango de valores
plt.title('Distribució de Consitència amb les Regles entre Models')
plt.ylabel('Consitència')
plt.xlabel('Models')
plt.grid(axis='y', linestyle='--', alpha=0.7)


plt.savefig(os.getcwd() + "/archivos_informe/grafico_reglas_distribucion.jpg")