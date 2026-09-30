"""
2_entrenar_modelo.py
---------------------
Entrena un clasificador binario (barco / no barco) usando transfer learning:

  1. Usa MobileNetV2 (preentrenada en ImageNet, sin la capa final) como
     extractor de caracteristicas -> convierte cada imagen en un vector.
  2. Con esos vectores entrena un clasificador clasico (Regresion Logistica)
     que funciona bien incluso con pocas imagenes (80 es poco para una CNN
     desde cero, pero es suficiente para esto).
  3. Evalua con un conjunto de prueba y muestra:
       - Accuracy, Precision, Recall, F1
       - Matriz de confusion (se guarda como imagen y se muestra en pantalla)
  4. Guarda el modelo entrenado en modelo_barcos.joblib para usarlo despues
     en 3_predecir_imagenes_nuevas.py

Requiere que ya hayas corrido 1_etiquetar_imagenes.py y tengas labels.csv

Uso:
    python 2_entrenar_modelo.py
"""

import os
import csv
import numpy as np
import joblib
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

CARPETA_IMAGENES = "images/train"
ARCHIVO_LABELS = "labels.csv"
MODELO_SALIDA = "modelo_barcos.joblib"
TAMANO_IMG = (224, 224)


def cargar_labels():
    filas = []
    with open(ARCHIVO_LABELS, newline="", encoding="utf-8") as f:
        lector = csv.DictReader(f)
        for fila in lector:
            filas.append((fila["imagen"], int(fila["label"])))
    return filas


def cargar_y_preprocesar_imagen(ruta):
    img = Image.open(ruta).convert("RGB").resize(TAMANO_IMG)
    arr = np.array(img).astype("float32")
    return arr


def extraer_caracteristicas(rutas, extractor):
    """Convierte una lista de imagenes en una matriz de vectores de caracteristicas."""
    lote = np.stack([cargar_y_preprocesar_imagen(r) for r in rutas])
    lote = preprocess_input(lote)
    features = extractor.predict(lote, verbose=0)
    return features


def main():
    if not os.path.exists(ARCHIVO_LABELS):
        print(f"No encuentro '{ARCHIVO_LABELS}'. Corre primero 1_etiquetar_imagenes.py")
        return

    datos = cargar_labels()
    print(f"Imagenes etiquetadas encontradas: {len(datos)}")

    n_barco = sum(1 for _, l in datos if l == 1)
    n_no_barco = sum(1 for _, l in datos if l == 0)
    print(f"  - Con barco:    {n_barco}")
    print(f"  - Sin barco:    {n_no_barco}")

    if len(datos) < 20:
        print("\nAdvertencia: tienes muy pocas imagenes etiquetadas. "
              "Idealmente etiqueta las 80 antes de entrenar.")

    rutas = [os.path.join(CARPETA_IMAGENES, nombre) for nombre, _ in datos]
    y = np.array([label for _, label in datos])

    print("\nCargando MobileNetV2 (puede tardar un poco la primera vez, "
          "descarga los pesos preentrenados)...")
    extractor = MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet", pooling="avg"
    )

    print("Extrayendo caracteristicas de las imagenes...")
    X = extraer_caracteristicas(rutas, extractor)
    print(f"Shape de las caracteristicas: {X.shape}")

    # Separamos train / test. Con datasets chicos usamos stratify para
    # mantener la proporcion de barco/no-barco en ambos conjuntos.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"\nEntrenamiento: {len(y_train)} imagenes | Prueba: {len(y_test)} imagenes")

    # Pipeline: escalado + clasificador. Probamos LogisticRegression, que
    # suele funcionar muy bien sobre features de una CNN preentrenada.
    modelo = Pipeline([
        ("escalador", StandardScaler()),
        ("clasificador", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])

    # Validacion cruzada sobre el set de entrenamiento para tener una idea
    # mas robusta del desempeño (util cuando hay pocos datos).
    if len(y_train) >= 10:
        cv_scores = cross_val_score(modelo, X_train, y_train, cv=min(5, len(y_train)))
        print(f"Accuracy validacion cruzada (train): {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    print("\n=== Resultados sobre el conjunto de prueba ===")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.3f}")
    print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.3f}")
    print(f"F1-score:  {f1_score(y_test, y_pred, zero_division=0):.3f}")
    print("\nReporte de clasificacion:")
    print(classification_report(y_test, y_pred, target_names=["No barco", "Barco"], zero_division=0))

    # Matriz de confusion
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No barco", "Barco"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Matriz de confusion - Deteccion de barcos")
    plt.tight_layout()
    plt.savefig("matriz_confusion.png", dpi=150)
    print("\nMatriz de confusion guardada como 'matriz_confusion.png'")
    plt.show()

    # Guardamos el modelo entrenado (el pipeline: escalador + clasificador)
    joblib.dump(modelo, MODELO_SALIDA)
    print(f"\nModelo guardado en '{MODELO_SALIDA}'")
    print("Ya puedes correr 3_predecir_imagenes_nuevas.py")


if __name__ == "__main__":
    main()
