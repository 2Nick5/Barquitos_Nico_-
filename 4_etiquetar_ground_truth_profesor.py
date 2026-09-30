"""
4_etiquetar_ground_truth_profesor.py
--------------------------------------
Igual que 1_etiquetar_imagenes.py, pero apunta a la carpeta de las
imagenes que trae el profesor (images/predict/) y guarda tu etiquetado
en un archivo SEPARADO: ground_truth_profesor.csv

Este archivo es la "verdad" contra la que despues vas a comparar lo que
prediga el modelo (paso 5). Etiqueta estas imagenes usando el mismo
criterio que usaste para las 80 de entrenamiento -> ver criterios_etiquetado.md

Controles:
    y  -> SI hay barco
    n  -> NO hay barco
    s  -> saltar esta imagen (la revisas despues)
    q  -> guardar y salir

Uso:
    python 4_etiquetar_ground_truth_profesor.py
"""

import os
import csv
import matplotlib.pyplot as plt
from PIL import Image

CARPETA_IMAGENES = "images/predict"
ARCHIVO_LABELS = "ground_truth_profesor.csv"

EXTENSIONES_VALIDAS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def cargar_labels_existentes():
    etiquetadas = {}
    if os.path.exists(ARCHIVO_LABELS):
        with open(ARCHIVO_LABELS, newline="", encoding="utf-8") as f:
            lector = csv.DictReader(f)
            for fila in lector:
                etiquetadas[fila["imagen"]] = int(fila["label"])
    return etiquetadas


def guardar_label(imagen, label):
    existe = os.path.exists(ARCHIVO_LABELS)
    with open(ARCHIVO_LABELS, "a", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        if not existe:
            escritor.writerow(["imagen", "label"])
        escritor.writerow([imagen, label])


def main():
    if not os.path.isdir(CARPETA_IMAGENES):
        print(f"No encuentro la carpeta '{CARPETA_IMAGENES}'.")
        print("Pon ahi las 40 imagenes que te dio el profesor.")
        return

    todas = sorted(
        f for f in os.listdir(CARPETA_IMAGENES) if f.lower().endswith(EXTENSIONES_VALIDAS)
    )
    if not todas:
        print(f"No hay imagenes en '{CARPETA_IMAGENES}'.")
        return

    ya_etiquetadas = cargar_labels_existentes()
    pendientes = [f for f in todas if f not in ya_etiquetadas]

    print(f"Total de imagenes del profesor: {len(todas)}")
    print(f"Ya etiquetadas (ground truth): {len(ya_etiquetadas)}")
    print(f"Pendientes: {len(pendientes)}")
    print("\nRecuerda usar el mismo criterio que en criterios_etiquetado.md\n")

    if not pendientes:
        print("Ya etiquetaste todas. Puedes correr 5_evaluar_resultados_profesor.py")
        return

    print("Controles: y = SI barco | n = NO barco | s = saltar | q = guardar y salir\n")

    fig, ax = plt.subplots(figsize=(7, 7))
    idx = {"i": 0}

    def mostrar_imagen():
        ax.clear()
        nombre = pendientes[idx["i"]]
        ruta = os.path.join(CARPETA_IMAGENES, nombre)
        img = Image.open(ruta).convert("RGB")
        ax.imshow(img)
        ax.set_title(
            f"[GROUND TRUTH] [{idx['i']+1}/{len(pendientes)}] {nombre}\n"
            f"y = barco   |   n = no barco   |   s = saltar   |   q = salir"
        )
        ax.axis("off")
        fig.canvas.draw()

    def on_key(event):
        nombre = pendientes[idx["i"]]

        if event.key == "y":
            guardar_label(nombre, 1)
            idx["i"] += 1
        elif event.key == "n":
            guardar_label(nombre, 0)
            idx["i"] += 1
        elif event.key == "s":
            idx["i"] += 1
        elif event.key == "q":
            plt.close(fig)
            return

        if idx["i"] >= len(pendientes):
            print("\nTerminaste de etiquetar el ground truth del profesor.")
            plt.close(fig)
            return

        mostrar_imagen()

    fig.canvas.mpl_connect("key_press_event", on_key)
    mostrar_imagen()
    plt.show()

    print(f"\nListo. Ground truth guardado en '{ARCHIVO_LABELS}'.")
    print("Ahora corre: python 5_evaluar_resultados_profesor.py")


if __name__ == "__main__":
    main()
