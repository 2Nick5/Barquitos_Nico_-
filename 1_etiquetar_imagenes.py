"""
1_etiquetar_imagenes.py
------------------------
Recorre todas las imagenes en images/train/ y te pregunta,
imagen por imagen, si hay un barco o no.

Controles (con la ventana de la imagen activa):
    y  -> SI hay barco
    n  -> NO hay barco
    s  -> saltar esta imagen (la revisas despues)
    q  -> guardar y salir

Los resultados se van guardando en labels.csv de forma incremental,
asi que si cierras a la mitad, la proxima vez que corras el script
retoma donde ibas (no te vuelve a preguntar por las que ya etiquetaste).

Uso:
    python 1_etiquetar_imagenes.py
"""

import os
import csv
import matplotlib.pyplot as plt
from PIL import Image

CARPETA_IMAGENES = "images/train"
ARCHIVO_LABELS = "labels.csv"

EXTENSIONES_VALIDAS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def cargar_labels_existentes():
    """Lee labels.csv si ya existe, para no repreguntar imagenes ya etiquetadas."""
    etiquetadas = {}
    if os.path.exists(ARCHIVO_LABELS):
        with open(ARCHIVO_LABELS, newline="", encoding="utf-8") as f:
            lector = csv.DictReader(f)
            for fila in lector:
                etiquetadas[fila["imagen"]] = int(fila["label"])
    return etiquetadas


def guardar_label(imagen, label):
    """Agrega una fila al CSV (crea el header si el archivo no existe)."""
    existe = os.path.exists(ARCHIVO_LABELS)
    with open(ARCHIVO_LABELS, "a", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        if not existe:
            escritor.writerow(["imagen", "label"])
        escritor.writerow([imagen, label])


def main():
    if not os.path.isdir(CARPETA_IMAGENES):
        print(f"No encuentro la carpeta '{CARPETA_IMAGENES}'.")
        print("Crea esa carpeta y pon ahi tus 80 imagenes satelitales.")
        return

    todas = sorted(
        f for f in os.listdir(CARPETA_IMAGENES) if f.lower().endswith(EXTENSIONES_VALIDAS)
    )
    if not todas:
        print(f"No hay imagenes en '{CARPETA_IMAGENES}'.")
        return

    ya_etiquetadas = cargar_labels_existentes()
    pendientes = [f for f in todas if f not in ya_etiquetadas]

    print(f"Total de imagenes encontradas: {len(todas)}")
    print(f"Ya etiquetadas: {len(ya_etiquetadas)}")
    print(f"Pendientes: {len(pendientes)}")
    if not pendientes:
        print("No hay nada pendiente. Puedes correr 2_entrenar_modelo.py")
        return

    print("\nControles: y = SI barco | n = NO barco | s = saltar | q = guardar y salir\n")

    fig, ax = plt.subplots(figsize=(7, 7))

    idx = {"i": 0}  # usamos un dict para poder modificarlo dentro del callback

    def mostrar_imagen():
        ax.clear()
        nombre = pendientes[idx["i"]]
        ruta = os.path.join(CARPETA_IMAGENES, nombre)
        img = Image.open(ruta).convert("RGB")
        ax.imshow(img)
        ax.set_title(
            f"[{idx['i']+1}/{len(pendientes)}] {nombre}\n"
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
            print("\nTerminaste de revisar todas las imagenes pendientes.")
            plt.close(fig)
            return

        mostrar_imagen()

    fig.canvas.mpl_connect("key_press_event", on_key)
    mostrar_imagen()
    plt.show()

    print(f"\nListo. Resultados guardados en '{ARCHIVO_LABELS}'.")


if __name__ == "__main__":
    main()
