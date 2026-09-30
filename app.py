"""
-------
Aplicacion para todo el flujo de deteccion de
barcos en imagenes satelitales:

    1) Etiquetar tus imagenes de entrenamiento.
    2) Entrenar el modelo (MobileNetV2 + Regresion Logistica) con
       metricas de validacion cruzada en vivo.
    3) Predecir imagenes nuevas seleccionando otra
       carpeta, con opcion de cargar el ground truth real y ver
       accuracy / precision / recall / matriz de confusion.

Como correrla:
    pip install -r requirements.txt
    streamlit run app.py

Se abre en el navegador (http://localhost:8501).

"""

import os
import time
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
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

# ----------------------------------------------------------------------
# Configuracion general
# ----------------------------------------------------------------------

EXTENSIONES_VALIDAS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
TAMANO_IMG = (224, 224)
ARCHIVO_LABELS_TRAIN = "labels_train.csv"
ARCHIVO_GT_PREDICT = "ground_truth_predict.csv"
ARCHIVO_MODELO = "modelo_barcos.joblib"

st.set_page_config(page_title="Detector de Barcos Satelital", page_icon="🚢", layout="wide")


# ----------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------

def elegir_carpeta():
    """Abre el explorador de carpetas nativo del sistema operativo."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        carpeta = filedialog.askdirectory()
        root.destroy()
        return carpeta
    except Exception:
        return None


def listar_imagenes(carpeta):
    if not carpeta or not os.path.isdir(carpeta):
        return []
    return sorted(f for f in os.listdir(carpeta) if f.lower().endswith(EXTENSIONES_VALIDAS))


def guardar_labels(dic, archivo):
    pd.DataFrame(
        [{"imagen": k, "label": v} for k, v in dic.items()]
    ).to_csv(archivo, index=False)


def cargar_labels(archivo):
    if os.path.exists(archivo):
        df = pd.read_csv(archivo)
        return dict(zip(df["imagen"], df["label"].astype(int)))
    return {}


@st.cache_resource(show_spinner="Cargando MobileNetV2 (solo la primera vez)...")
def cargar_extractor():
    return MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet", pooling="avg"
    )


def cargar_imagen_np(ruta):
    img = Image.open(ruta).convert("RGB").resize(TAMANO_IMG)
    return np.array(img).astype("float32")


def extraer_features(rutas, extractor, progreso_texto="Extrayendo caracteristicas"):
    barra = st.progress(0.0, text=progreso_texto)
    feats = []
    for i, ruta in enumerate(rutas):
        arr = cargar_imagen_np(ruta)
        arr = preprocess_input(np.expand_dims(arr, 0))
        f = extractor.predict(arr, verbose=0)[0]
        feats.append(f)
        barra.progress((i + 1) / len(rutas), text=f"{progreso_texto} ({i+1}/{len(rutas)})")
    barra.empty()
    return np.array(feats)


def graficar_matriz_confusion(y_true, y_pred, titulo):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(cm, display_labels=["No barco", "Barco"]).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(titulo)
    plt.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Estado inicial de la sesion
# ----------------------------------------------------------------------

def init_state():
    defaults = {
        "train_folder": "",
        "predict_folder": "",
        "labels_train": cargar_labels(ARCHIVO_LABELS_TRAIN),
        "labels_gt": cargar_labels(ARCHIVO_GT_PREDICT),
        "modelo": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if st.session_state["modelo"] is None and os.path.exists(ARCHIVO_MODELO):
        try:
            st.session_state["modelo"] = joblib.load(ARCHIVO_MODELO)
        except Exception:
            pass


init_state()

st.title("🚢 Detector de Barcos en Imágenes Satelitales")

# Barra lateral con estado general
with st.sidebar:
    st.header("Estado del proyecto")
    st.metric("Imágenes etiquetadas (train)", len(st.session_state["labels_train"]))
    st.metric("Modelo entrenado", "Sí ✅" if st.session_state["modelo"] is not None else "No ❌")
    st.metric("Ground truth cargado (predicción)", len(st.session_state["labels_gt"]))
    st.divider()
    st.caption(
        "Flujo: 1) Etiqueta tus imágenes → 2) Entrena el modelo → "
        "3) Predice imágenes nuevas (y compara contra ground truth si lo tienes)."
    )

tab1, tab2, tab3 = st.tabs(
    ["1️⃣ Etiquetar entrenamiento", "2️⃣ Entrenar modelo", "3️⃣ Predecir y evaluar"]
)

# ----------------------------------------------------------------------
# TAB 1: Etiquetar imagenes de entrenamiento
# ----------------------------------------------------------------------

with tab1:
    st.subheader("Selecciona la carpeta con tus imágenes de entrenamiento")

    col_btn, col_txt = st.columns([1, 3])
    with col_btn:
        if st.button("📁 Seleccionar carpeta", key="btn_train_folder"):
            carpeta = elegir_carpeta()
            if carpeta:
                st.session_state["train_folder"] = carpeta
                st.rerun()
    with col_txt:
        st.session_state["train_folder"] = st.text_input(
            "Ruta de la carpeta (se llena sola al usar el botón, o escríbela tú):",
            value=st.session_state["train_folder"],
            key="txt_train_folder",
        )

    carpeta = st.session_state["train_folder"]
    imagenes = listar_imagenes(carpeta)

    if carpeta and not imagenes:
        st.warning("No se encontraron imágenes válidas en esa carpeta.")
    elif imagenes:
        labels = st.session_state["labels_train"]
        pendientes = [f for f in imagenes if f not in labels]

        etiquetadas = len(labels)
        total = len(imagenes)
        st.progress(etiquetadas / total if total else 0, text=f"{etiquetadas}/{total} etiquetadas")

        n_barco = sum(1 for v in labels.values() if v == 1)
        n_no_barco = sum(1 for v in labels.values() if v == 0)
        c1, c2 = st.columns(2)
        c1.metric("Con barco", n_barco)
        c2.metric("Sin barco", n_no_barco)

        if pendientes:
            actual = pendientes[0]
            ruta = os.path.join(carpeta, actual)

            col_img, col_btns = st.columns([2, 1])
            with col_img:
                st.image(ruta, caption=actual, width=400)
            with col_btns:
                st.write("¿Hay un barco en esta imagen?")
                if st.button("✅ Sí, hay barco", use_container_width=True, key="lbl_si"):
                    labels[actual] = 1
                    guardar_labels(labels, ARCHIVO_LABELS_TRAIN)
                    st.rerun()
                if st.button("❌ No hay barco", use_container_width=True, key="lbl_no"):
                    labels[actual] = 0
                    guardar_labels(labels, ARCHIVO_LABELS_TRAIN)
                    st.rerun()
                if st.button("⏭️ Saltar por ahora", use_container_width=True, key="lbl_skip"):
                    pendientes.append(pendientes.pop(0))
                    st.rerun()
        else:
            st.success("¡Etiquetaste todas las imágenes de esta carpeta! Ve a la pestaña 2.")
    else:
        st.info("Selecciona una carpeta para comenzar a etiquetar.")

# ----------------------------------------------------------------------
# TAB 2: Entrenar modelo
# ----------------------------------------------------------------------

with tab2:
    st.subheader("Entrenar el modelo")

    labels = st.session_state["labels_train"]
    n_etiquetadas = len(labels)

    if n_etiquetadas < 10:
        st.warning(
            f"Solo tienes {n_etiquetadas} imágenes etiquetadas. "
            "Etiqueta al menos ~20 (idealmente tus 80) en la pestaña 1 antes de entrenar."
        )
    else:
        st.write(f"Imágenes disponibles para entrenar: **{n_etiquetadas}**")

        if st.button("🚀 Entrenar modelo", type="primary"):
            carpeta = st.session_state["train_folder"]
            nombres = list(labels.keys())
            rutas = [os.path.join(carpeta, n) for n in nombres]
            y = np.array([labels[n] for n in nombres])

            extractor = cargar_extractor()
            X = extraer_features(rutas, extractor, "Extrayendo características")

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y
            )

            # --- Validacion cruzada EN VIVO ---
            st.markdown("#### Validación cruzada (en vivo)")
            chart_ph = st.empty()
            metric_ph = st.empty()

            n_splits = min(5, np.bincount(y_train).min()) if len(np.unique(y_train)) > 1 else 2
            n_splits = max(2, n_splits)
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

            scores = []
            for fold, (idx_tr, idx_val) in enumerate(skf.split(X_train, y_train)):
                pipe = Pipeline([
                    ("escalador", StandardScaler()),
                    ("clasificador", LogisticRegression(max_iter=2000, class_weight="balanced")),
                ])
                pipe.fit(X_train[idx_tr], y_train[idx_tr])
                acc = pipe.score(X_train[idx_val], y_train[idx_val])
                scores.append(acc)

                chart_ph.line_chart(pd.DataFrame({"accuracy por fold": scores}))
                metric_ph.metric("Accuracy promedio (validación cruzada)", f"{np.mean(scores):.2%}")
                time.sleep(0.25)

            # --- Modelo final ---
            modelo_final = Pipeline([
                ("escalador", StandardScaler()),
                ("clasificador", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ])
            modelo_final.fit(X_train, y_train)
            y_pred = modelo_final.predict(X_test)

            st.markdown("#### Resultados sobre el conjunto de prueba (holdout)")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
            c2.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.2%}")
            c3.metric("Recall", f"{recall_score(y_test, y_pred, zero_division=0):.2%}")
            c4.metric("F1-score", f"{f1_score(y_test, y_pred, zero_division=0):.2%}")

            col_cm, col_rep = st.columns(2)
            with col_cm:
                fig = graficar_matriz_confusion(y_test, y_pred, "Matriz de confusión (holdout)")
                st.pyplot(fig)
            with col_rep:
                st.text("Reporte de clasificación:")
                st.text(
                    classification_report(
                        y_test, y_pred, target_names=["No barco", "Barco"], zero_division=0
                    )
                )

            joblib.dump(modelo_final, ARCHIVO_MODELO)
            st.session_state["modelo"] = modelo_final
            st.success(f"Modelo entrenado y guardado en '{ARCHIVO_MODELO}'. Ve a la pestaña 3.")

# ----------------------------------------------------------------------
# TAB 3: Predecir y evaluar (imagenes del profesor)
# ----------------------------------------------------------------------

with tab3:
    st.subheader("Predecir imágenes nuevas")

    if st.session_state["modelo"] is None:
        st.warning("Primero entrena el modelo en la pestaña 2.")
    else:
        col_btn, col_txt = st.columns([1, 3])
        with col_btn:
            if st.button("📁 Seleccionar carpeta", key="btn_predict_folder"):
                carpeta = elegir_carpeta()
                if carpeta:
                    st.session_state["predict_folder"] = carpeta
                    st.rerun()
        with col_txt:
            st.session_state["predict_folder"] = st.text_input(
                "Ruta de la carpeta con las imágenes a predecir:",
                value=st.session_state["predict_folder"],
                key="txt_predict_folder",
            )

        carpeta_p = st.session_state["predict_folder"]
        imagenes_p = listar_imagenes(carpeta_p)

        if carpeta_p and not imagenes_p:
            st.warning("No se encontraron imágenes válidas en esa carpeta.")
        elif imagenes_p:
            st.write(f"Imágenes encontradas: **{len(imagenes_p)}**")

            tiene_gt = st.checkbox(
                "Tengo (o voy a poner) las etiquetas reales (ground truth) de estas imágenes, "
                "para medir el desempeño real del modelo",
            )

            if tiene_gt:
                labels_gt = st.session_state["labels_gt"]
                pendientes_gt = [f for f in imagenes_p if f not in labels_gt]

                st.markdown("##### Etiquetado de ground truth")
                st.progress(
                    (len(imagenes_p) - len(pendientes_gt)) / len(imagenes_p),
                    text=f"{len(imagenes_p) - len(pendientes_gt)}/{len(imagenes_p)} etiquetadas",
                )

                if pendientes_gt:
                    actual = pendientes_gt[0]
                    ruta = os.path.join(carpeta_p, actual)
                    col_img, col_btns = st.columns([2, 1])
                    with col_img:
                        st.image(ruta, caption=actual, width=350)
                    with col_btns:
                        st.write("¿Hay un barco en esta imagen (verdad real)?")
                        if st.button("✅ Sí hay barco", key="gt_si"):
                            labels_gt[actual] = 1
                            guardar_labels(labels_gt, ARCHIVO_GT_PREDICT)
                            st.rerun()
                        if st.button("❌ No hay barco", key="gt_no"):
                            labels_gt[actual] = 0
                            guardar_labels(labels_gt, ARCHIVO_GT_PREDICT)
                            st.rerun()
                else:
                    st.success("Ground truth completo para esta carpeta.")

            velocidad = st.slider(
                "Velocidad de revelado en vivo (segundos entre imagen)", 0.0, 1.0, 0.2, 0.05
            )

            if st.button("🔍 Predecir", type="primary"):
                extractor = cargar_extractor()
                rutas_p = [os.path.join(carpeta_p, n) for n in imagenes_p]
                X_p = extraer_features(rutas_p, extractor, "Extrayendo características")

                modelo = st.session_state["modelo"]
                y_pred_all = modelo.predict(X_p)
                probas_all = modelo.predict_proba(X_p)[:, 1]

                labels_gt = st.session_state["labels_gt"]
                usa_gt = tiene_gt and all(n in labels_gt for n in imagenes_p)
                if tiene_gt and not usa_gt:
                    st.info(
                        "Aún no has etiquetado todo el ground truth, así que las métricas en "
                        "vivo no se mostrarán (pero sí las predicciones)."
                    )

                st.markdown("#### Resultados en vivo")
                img_ph = st.empty()
                metrics_ph = st.empty()
                chart_ph = st.empty()
                cm_ph = st.empty()

                resultados = []
                y_true_parcial, y_pred_parcial = [], []

                for i, nombre in enumerate(imagenes_p):
                    pred = int(y_pred_all[i])
                    proba = float(probas_all[i])
                    real = labels_gt.get(nombre) if usa_gt else None

                    resultados.append(
                        {"imagen": nombre, "prediccion": pred, "probabilidad_barco": proba, "real": real}
                    )

                    with img_ph.container():
                        c1, c2 = st.columns([1, 3])
                        with c1:
                            st.image(os.path.join(carpeta_p, nombre), width=150)
                        with c2:
                            st.write(f"**{nombre}**")
                            st.write(
                                f"Predicción: {'🚢 Barco' if pred == 1 else '🌊 No barco'} "
                                f"({proba:.1%} confianza)"
                            )
                            if real is not None:
                                st.write(f"Real: {'🚢 Barco' if real == 1 else '🌊 No barco'}")
                                st.write("✅ Correcto" if real == pred else "❌ Incorrecto")

                    if usa_gt:
                        y_true_parcial.append(real)
                        y_pred_parcial.append(pred)

                        acc = accuracy_score(y_true_parcial, y_pred_parcial)
                        prec = precision_score(y_true_parcial, y_pred_parcial, zero_division=0)
                        rec = recall_score(y_true_parcial, y_pred_parcial, zero_division=0)

                        with metrics_ph.container():
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("Accuracy (en vivo)", f"{acc:.1%}")
                            m2.metric("Precision", f"{prec:.1%}")
                            m3.metric("Recall", f"{rec:.1%}")
                            m4.metric("Procesadas", f"{i+1}/{len(imagenes_p)}")

                        cum_acc = [
                            accuracy_score(y_true_parcial[: j + 1], y_pred_parcial[: j + 1])
                            for j in range(len(y_true_parcial))
                        ]
                        chart_ph.line_chart(pd.DataFrame({"accuracy acumulada": cum_acc}))

                        fig = graficar_matriz_confusion(
                            y_true_parcial, y_pred_parcial, "Matriz de confusión (en vivo)"
                        )
                        cm_ph.pyplot(fig)
                        plt.close(fig)

                    time.sleep(velocidad)

                img_ph.empty()

                st.markdown("#### Resultados finales")
                df_resultados = pd.DataFrame(resultados)
                df_resultados["prediccion_texto"] = df_resultados["prediccion"].map(
                    {1: "Barco", 0: "No barco"}
                )
                if usa_gt:
                    df_resultados["real_texto"] = df_resultados["real"].map({1: "Barco", 0: "No barco"})
                    df_resultados["acierto"] = df_resultados["real"] == df_resultados["prediccion"]

                st.dataframe(df_resultados, use_container_width=True)

                st.download_button(
                    "⬇️ Descargar resultados en CSV",
                    df_resultados.to_csv(index=False).encode("utf-8"),
                    file_name="resultados_prediccion.csv",
                    mime="text/csv",
                )

                if usa_gt:
                    st.markdown("#### Métricas finales (ground truth completo)")
                    y_t = df_resultados["real"].tolist()
                    y_p = df_resultados["prediccion"].tolist()
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Accuracy", f"{accuracy_score(y_t, y_p):.2%}")
                    c2.metric("Precision", f"{precision_score(y_t, y_p, zero_division=0):.2%}")
                    c3.metric("Recall", f"{recall_score(y_t, y_p, zero_division=0):.2%}")
                    c4.metric("F1-score", f"{f1_score(y_t, y_p, zero_division=0):.2%}")

                    fig_final = graficar_matriz_confusion(
                        y_t, y_p, "Matriz de confusión final - Imágenes del profesor"
                    )
                    st.pyplot(fig_final)
        else:
            st.info("Selecciona una carpeta con las imágenes a predecir.")
