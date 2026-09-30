"""
3_predecir_imagenes_nuevas.py
-------------------------------
Carga el modelo entrenado (modelo_barcos.joblib) y lo usa para predecir,
imagen por imagen, si hay o no un barco en cada una de las imagenes que
pongas en images/predict/ (tus 40 imagenes externas).

Genera:
  - predicciones.csv        -> tabla con imagen, prediccion y probabilidad
  - grid_predicciones.png   -> collage con todas las imagenes y su etiqueta

Uso:
    python 3_predecir_imagenes_nuevas.py
"""

import os
import csv
import math
import numpy as np
import joblib
import matplotlib.pyplot as plt

from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

CARPETA_IMAGENES = "images/predict"
MODELO = "modelo_barcos.joblib"
SALIDA_CSV = "predicciones.csv"
SALIDA_GRID = "grid_predicciones.png"
TAMANO_IMG = (224, 224)
EXTENSIONES_VALIDAS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def cargar_y_preprocesar_imagen(ruta):
    img = Image.open(ruta).convert("RGB").resize(TAMANO_IMG)
    arr = np.array(img).astype("float32")
    return arr


def main():
    if not os.path.exists(MODELO):
        print(f"No encuentro '{MODELO}'. Corre primero 2_entrenar_modelo.py")
        return

    if not os.path.isdir(CARPETA_IMAGENES):
        print(f"No encuentro la carpeta '{CARPETA_IMAGENES}'.")
        print("Crea esa carpeta y pon ahi tus 40 imagenes externas.")
        return

    nombres = sorted(
        f for f in os.listdir(CARPETA_IMAGENES) if f.lower().endswith(EXTENSIONES_VALIDAS)
    )
    if not nombres:
        print(f"No hay imagenes en '{CARPETA_IMAGENES}'.")
        return

    print(f"Imagenes a predecir: {len(nombres)}")

    print("Cargando MobileNetV2...")
    extractor = MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet", pooling="avg"
    )

    print("Cargando modelo entrenado...")
    modelo = joblib.load(MODELO)

    rutas = [os.path.join(CARPETA_IMAGENES, n) for n in nombres]
    lote = np.stack([cargar_y_preprocesar_imagen(r) for r in rutas])
    lote = preprocess_input(lote)

    print("Extrayendo caracteristicas...")
    X = extractor.predict(lote, verbose=0)

    print("Prediciendo...")
    preds = modelo.predict(X)
    probas = modelo.predict_proba(X)[:, 1]  # probabilidad de "barco"

    # Guardar CSV
    with open(SALIDA_CSV, "w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerow(["imagen", "prediccion", "probabilidad_barco"])
        for nombre, pred, proba in zip(nombres, preds, probas):
            etiqueta = "Barco" if pred == 1 else "No barco"
            escritor.writerow([nombre, etiqueta, f"{proba:.3f}"])

    print(f"\nResultados guardados en '{SALIDA_CSV}'")

    n_barco = int(preds.sum())
    print(f"Total detectados como BARCO:    {n_barco}")
    print(f"Total detectados como NO BARCO: {len(preds) - n_barco}")

    # Collage visual con todas las predicciones
    n = len(nombres)
    cols = 5
    filas = math.ceil(n / cols)
    fig, axes = plt.subplots(filas, cols, figsize=(cols * 3, filas * 3))
    axes = np.array(axes).reshape(-1)

    for i, (ruta, pred, proba) in enumerate(zip(rutas, preds, probas)):
        img = Image.open(ruta).convert("RGB")
        axes[i].imshow(img)
        etiqueta = "BARCO" if pred == 1 else "no barco"
        color = "green" if pred == 1 else "red"
        axes[i].set_title(f"{etiqueta}\n({proba:.2f})", color=color, fontsize=9)
        axes[i].axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(SALIDA_GRID, dpi=150)
    print(f"Collage de resultados guardado en '{SALIDA_GRID}'")
    plt.show()


if __name__ == "__main__":
    main()
