"""
5_evaluar_resultados_profesor.py
-----------------------------------
Este es el script que te da los resultados "reales" para mostrarle al
profesor: compara lo que PREDICE el modelo contra el GROUND TRUTH que
tu etiquetaste manualmente (paso 4), sobre las 40 imagenes que el
profesor trajo.

Requiere que ya existan:
  - modelo_barcos.joblib             (paso 2: entrenar el modelo)
  - ground_truth_profesor.csv        (paso 4: tu etiquetado manual)
  - images/predict/                  (las 40 imagenes del profesor)

Genera:
  - matriz_confusion_profesor.png    -> matriz de confusion final
  - comparacion_profesor.csv         -> tabla con imagen, etiqueta real,
                                          prediccion del modelo, y si acerto o no

Uso:
    python 5_evaluar_resultados_profesor.py
"""

import os
import csv
import numpy as np
import joblib
import matplotlib.pyplot as plt

from PIL import Image
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

CARPETA_IMAGENES = "images/predict"
MODELO = "modelo_barcos.joblib"
GROUND_TRUTH = "ground_truth_profesor.csv"
SALIDA_CSV = "comparacion_profesor.csv"
SALIDA_MATRIZ = "matriz_confusion_profesor.png"
TAMANO_IMG = (224, 224)


def cargar_ground_truth():
    etiquetas = {}
    with open(GROUND_TRUTH, newline="", encoding="utf-8") as f:
        lector = csv.DictReader(f)
        for fila in lector:
            etiquetas[fila["imagen"]] = int(fila["label"])
    return etiquetas


def cargar_y_preprocesar_imagen(ruta):
    img = Image.open(ruta).convert("RGB").resize(TAMANO_IMG)
    return np.array(img).astype("float32")


def main():
    if not os.path.exists(MODELO):
        print(f"No encuentro '{MODELO}'. Corre primero 2_entrenar_modelo.py")
        return

    if not os.path.exists(GROUND_TRUTH):
        print(f"No encuentro '{GROUND_TRUTH}'.")
        print("Corre primero 4_etiquetar_ground_truth_profesor.py para etiquetar "
              "manualmente las 40 imagenes del profesor.")
        return

    ground_truth = cargar_ground_truth()

    # Solo evaluamos las imagenes que SI tienen ground truth etiquetado
    nombres = sorted(ground_truth.keys())
    faltantes = [
        f for f in os.listdir(CARPETA_IMAGENES)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
        and f not in ground_truth
    ]

    if faltantes:
        print(f"Aviso: hay {len(faltantes)} imagenes en '{CARPETA_IMAGENES}' "
              f"que todavia NO tienen ground truth etiquetado. No se van a "
              f"incluir en la evaluacion:")
        for f in faltantes:
            print(f"   - {f}")
        print()

    if not nombres:
        print("No hay ninguna imagen con ground truth etiquetado. Nada que evaluar.")
        return

    print(f"Evaluando {len(nombres)} imagenes con ground truth...")

    print("Cargando MobileNetV2...")
    extractor = MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet", pooling="avg"
    )

    print("Cargando modelo entrenado...")
    modelo = joblib.load(MODELO)

    rutas = [os.path.join(CARPETA_IMAGENES, n) for n in nombres]
    lote = np.stack([cargar_y_preprocesar_imagen(r) for r in rutas])
    lote = preprocess_input(lote)

    print("Extrayendo caracteristicas y prediciendo...")
    X = extractor.predict(lote, verbose=0)
    y_pred = modelo.predict(X)
    y_true = np.array([ground_truth[n] for n in nombres])

    # ---- Metricas ----
    print("\n=== Resultados sobre las imagenes del profesor (ground truth real) ===")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.3f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"F1-score:  {f1_score(y_true, y_pred, zero_division=0):.3f}")
    print("\nReporte de clasificacion:")
    print(classification_report(y_true, y_pred, target_names=["No barco", "Barco"], zero_division=0))

    # ---- Matriz de confusion ----
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No barco", "Barco"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Matriz de confusion - Imagenes del profesor")
    plt.tight_layout()
    plt.savefig(SALIDA_MATRIZ, dpi=150)
    print(f"\nMatriz de confusion guardada en '{SALIDA_MATRIZ}'")
    plt.show()

    # ---- CSV detallado imagen por imagen ----
    with open(SALIDA_CSV, "w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerow(["imagen", "etiqueta_real", "prediccion_modelo", "acerto"])
        for nombre, real, pred in zip(nombres, y_true, y_pred):
            texto_real = "Barco" if real == 1 else "No barco"
            texto_pred = "Barco" if pred == 1 else "No barco"
            acerto = "SI" if real == pred else "NO"
            escritor.writerow([nombre, texto_real, texto_pred, acerto])

    print(f"Detalle imagen por imagen guardado en '{SALIDA_CSV}'")

    errores = [(n, r, p) for n, r, p in zip(nombres, y_true, y_pred) if r != p]
    if errores:
        print(f"\nImagenes donde el modelo se equivoco ({len(errores)}):")
        for nombre, real, pred in errores:
            print(f"   - {nombre}: real={'Barco' if real==1 else 'No barco'}, "
                  f"prediccion={'Barco' if pred==1 else 'No barco'}")
    else:
        print("\nEl modelo acerto en todas las imagenes evaluadas.")


if __name__ == "__main__":
    main()
