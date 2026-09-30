# Detector de barcos en imágenes satelitales

Clasificador binario (barco / no barco) usando transfer learning con
MobileNetV2 + un clasificador clásico entrenado sobre tus propias imágenes.

## 🌟 Opción recomendada: app.py (todo en una interfaz, un solo archivo)

En vez de correr los scripts sueltos por consola, `app.py` junta **todo**
(etiquetado, entrenamiento y predicción/evaluación) en una interfaz web
local con Streamlit, donde seleccionas las carpetas con el explorador de
archivos de tu sistema operativo y ves las métricas actualizándose en
tiempo real (accuracy, precision, recall, matriz de confusión).

```bash
pip install -r requirements.txt
streamlit run app.py
```

Se abre solo en tu navegador (`http://localhost:8501`). Tiene 3 pestañas:

1. **Etiquetar entrenamiento** — botón "📁 Seleccionar carpeta" (abre el
   explorador nativo de tu sistema) para elegir la carpeta con tus 80
   imágenes, y las vas etiquetando con botones "Sí hay barco" / "No hay
   barco" directamente en pantalla.
2. **Entrenar modelo** — un botón entrena el modelo, mostrando la
   validación cruzada actualizándose fold por fold en vivo, y al final
   muestra accuracy/precision/recall/F1 y la matriz de confusión.
3. **Predecir y evaluar** — seleccionas la carpeta con las 40 imágenes
   del profesor, opcionalmente marcas la casilla para etiquetar tú el
   ground truth ahí mismo, y al predecir se revela imagen por imagen con
   la accuracy, precision, recall y matriz de confusión **actualizándose
   en vivo** a medida que se procesa cada imagen. Al final puedes
   descargar los resultados en CSV.

---

## 1. Instalación

```bash
pip install -r requirements.txt
```

Requiere Python 3.9+ (recomendado). La primera vez que corras el
entrenamiento o la predicción, TensorFlow descargará automáticamente los
pesos de MobileNetV2 (unos ~14 MB), así que necesitas internet esa vez.

## 2. Organiza tus imágenes

```
barco_ia/
├── images/
│   ├── train/       <- pon aquí tus 80 imágenes a etiquetar
│   └── predict/     <- pon aquí tus 40 imágenes externas a predecir
```

## 3. Flujo de trabajo

### Paso 1 — Etiquetar tus 80 imágenes

```bash
python 1_etiquetar_imagenes.py
```

Genera `labels.csv` con dos columnas: `imagen,label` (1 = barco, 0 = no barco).
Puedes cerrar el script a la mitad y volver a correrlo; retoma donde ibas.

### Paso 2 — Entrenar el modelo

```bash
python 2_entrenar_modelo.py
```

- Extrae características de cada imagen con MobileNetV2 (preentrenada en
  ImageNet, sin volver a entrenarla — esto es clave para que funcione bien
  con pocas imágenes como 80).
- Entrena una Regresión Logística sobre esas características.
- Separa automáticamente ~75% para entrenar y ~25% para probar.
- Imprime accuracy, precision, recall, F1 y el reporte de clasificación.
- Muestra y guarda la **matriz de confusión** (`matriz_confusion.png`).
- Guarda el modelo entrenado en `modelo_barcos.joblib`.

### Paso 3 

```bash
python 3_predecir_imagenes_nuevas.py
```

- Carga el modelo entrenado.
- Predice barco / no barco para cada imagen en `images/predict/`.
- Genera `predicciones.csv` (imagen, predicción, probabilidad).
- Genera un collage visual `grid_predicciones.png` con todas las imágenes
  y su etiqueta predicha en verde (barco) o rojo (no barco).

### Paso 4 — Cuando el profesor traiga sus 40 fotos: etiquetar el ground truth

Antes de comparar el modelo contra la realidad, alguien tiene que decir cuál
es la verdad de cada imagen. Ese eres tú:

```bash
python 4_etiquetar_ground_truth_profesor.py
```

- Se pone las imágenes del profesor en `images/predict/`.
- Este script es igual al del Paso 1, pero guarda tu etiquetado en un
  archivo distinto: `ground_truth_profesor.csv`.
- **Importante**: usa el mismo criterio que usaste en el Paso 1 — revisa
  `criterios_etiquetado.md` para no ser inconsistente entre un set y otro.
- Este archivo NO se usa para entrenar el modelo, solo para poder medir
  qué tan bien predice sobre datos que el modelo nunca vio.

### Paso 5 — Comparar predicción del modelo vs. tu ground truth

```bash
python 5_evaluar_resultados_profesor.py
```

- Carga el modelo ya entrenado y predice las imágenes del profesor.
- Compara cada predicción contra `ground_truth_profesor.csv`.
- Imprime accuracy, precision, recall, F1 sobre estos datos "reales".
- Genera `matriz_confusion_profesor.png` — esta es la matriz que le
  muestras al profesor, porque refleja desempeño sobre datos externos
  que el modelo no vio en entrenamiento.
- Genera `comparacion_profesor.csv` con el detalle imagen por imagen
  (real vs. predicho vs. si acertó o no), y en la terminal te lista
  específicamente en cuáles se equivocó el modelo.
