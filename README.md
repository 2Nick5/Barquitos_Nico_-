# 🚢 Detector de Barcos en Imágenes Satelitales

Clasificador binario (barco / no barco) sobre imágenes satelitales, hecho
en Python con una interfaz web local (Streamlit). Permite etiquetar
imágenes de entrenamiento, entrenar el modelo y evaluar sobre imágenes
externas nuevas, todo con métricas y matriz de confusión en vivo.

---

**1. Extractor de características — MobileNetV2**
Red neuronal convolucional de Google, preentrenada sobre ImageNet (1.4M
de imágenes, 1000 clases). Se usa con `include_top=False` (sin su capa
final de clasificación) y `pooling="avg"`, de modo que cada imagen se
convierte en un vector de 1280 números que resume sus características
visuales (bordes, texturas, formas). Esta red **no se reentrena**, solo
se usa como calculadora de características fija.

**2. Clasificador — Regresión Logística**
Modelo de Machine Learning clásico (no red neuronal) de `scikit-learn`,
entrenado sobre esos vectores de 1280 números con tus imágenes
etiquetadas. Aprende a trazar una frontera entre los vectores de
"barco" y los de "no barco".

**¿Por qué así y no una CNN entrenada desde cero?**
Con un dataset de ~80-120 imágenes, entrenar una CNN completa desde cero
(millones de parámetros) produciría overfitting casi seguro — memorizaría
las fotos en vez de aprender el concepto general de "barco". Al usar
MobileNetV2 ya preentrenada, se aprovecha todo lo aprendido sobre formas
y texturas con ImageNet, y solo se entrena un clasificador pequeño que
necesita muy pocos datos para funcionar razonablemente bien.

### Sobre "épocas" y "learning rate"

La app tiene **dos modos de entrenamiento**, seleccionables con un
radio button en la pestaña "Entrenar modelo":

1. **Regresión Logística (rápido)** — el que se explicaba antes: no tiene
   épocas ni learning rate, converge sola con el método `lbfgs`.
2. **Red neuronal (Keras)** — una pequeña red densa (`Dense(128)` →
   `Dropout` → `Dense(64)` → `Dropout` → `Dense(1, sigmoid)`) entrenada
   **sobre las mismas características de 1280 números** que extrae
   MobileNetV2 (que sigue congelada). Esta red sí se entrena con
   descenso de gradiente real, así que **sí tiene**:
   - **Épocas** (`epochs`): cuántas veces la red ve el conjunto de
     entrenamiento completo. Configurable con un slider (5 a 150).
   - **Learning rate**: el tamaño del paso con el que el optimizador
     Adam ajusta los pesos en cada iteración. Configurable con un
     selector de valores típicos (0.1 a 0.00001).

   Mientras entrena, se ve la **curva de accuracy y de loss por época**
   actualizándose en vivo, tanto para el conjunto de entrenamiento como
   para el de validación — esto es lo que normalmente pide un profesor
   cuando habla de "épocas" y "learning rate".

**¿Cuál usar?** Con ~200 imágenes, el modo de red neuronal es razonable
y da curvas de entrenamiento genuinas. Si ves que la accuracy de
validación deja de mejorar (o empeora) mientras la de entrenamiento
sigue subiendo, es sobreajuste: baja el número de épocas o el learning
rate, o revisa el balance de tus clases.

---

## 🔧 Cómo mejorar el accuracy

Como no hay épocas/learning rate que ajustar, las palancas reales son
otras, en orden de impacto esperado:

1. **Más imágenes etiquetadas.** Pasar de ~80 a 150-200 imágenes suele
   ayudar más que cualquier ajuste técnico.
2. **Balance de clases.** Si hay muchas más imágenes de una clase que de
   otra (ej. 70 con barco vs. 10 sin barco), el modelo se sesga. Intenta
   mantener una proporción razonable.
3. **Consistencia en el etiquetado.** Usa siempre el mismo criterio (ver
   sección de criterios más abajo) tanto en las imágenes de entrenamiento
   como en las de evaluación.
4. **Regularización `C` de la Regresión Logística.** Es un hiperparámetro
   real y ajustable — en la pestaña "Entrenar modelo" de la app hay un
   control deslizante para `C`:
   - Valores bajos (ej. 0.1) → modelo más simple/general. Útil si notas
     que el modelo funciona muy bien en tus imágenes de entrenamiento
     pero mal en imágenes nuevas (señal de sobreajuste).
   - Valores altos (ej. 5-10) → el modelo se ajusta más a tus datos de
     entrenamiento. Puede ayudar si tienes muchos datos y buena calidad
     de etiquetado, pero con pocos datos aumenta el riesgo de sobreajuste.
   - Prueba distintos valores y compara el accuracy de validación
     cruzada y del holdout (los que se muestran al entrenar) para ver
     cuál generaliza mejor.
5. **Calidad/resolución de las imágenes.** Imágenes muy borrosas o de
   muy baja resolución le dan al extractor de características menos
   información útil.

---

## 📊 Cómo interpretar las métricas y gráficas

### Matriz de confusión

Aparece después de entrenar. El modelo se examina con un ~25% de las
imágenes que **no vio durante el entrenamiento** (conjunto de prueba /
holdout), para no hacer trampa. La matriz muestra, de esas imágenes de
examen, cuántas acertó como barco, cuántas acertó como no-barco, y en
cuáles se equivocó.

### Gráfica de validación cruzada

Es un diagnóstico interno durante el entrenamiento: divide el conjunto
de entrenamiento en varias partes (folds) y entrena/examina el modelo
varias veces con particiones distintas, mostrando el accuracy obtenido
en cada ronda. Si el accuracy es parecido en todas las rondas, el modelo
es consistente; si varía mucho, es señal de pocos datos o datos
desbalanceados. **No es la evaluación final**, solo un chequeo interno.

### Precision "en vivo" vs. final (pestaña de Predicción)

Al predecir las 40 imágenes externas, la app las revela una por una y
recalcula las métricas con las que lleva procesadas hasta el momento
("en vivo"). Es normal que salte mucho al principio. **La métrica que
importa es la final**, calculada con las 40 imágenes completas, mostrada
al final bajo "Métricas finales".

### Precision, Recall y F1

- **Precision:** de todas las veces que el modelo dijo "hay barco",
  ¿en cuántas acertó? Mide qué tan confiable es el modelo cuando dice
  que sí.
- **Recall:** de todos los barcos reales que había, ¿a cuántos detectó
  el modelo? Mide qué tanto se le escapa al modelo.
- **F1-score:** combina precision y recall en un solo número (promedio
  armónico). Un F1 alto solo se logra si ambas métricas son buenas al
  mismo tiempo — por eso es útil como resumen único.

---

## 🖥️ Cómo ejecutar la app

```bash
pip install -r requirements.txt
streamlit run app.py
```

Si Windows no reconoce `streamlit`, usa `python -m streamlit run app.py`.

Se abre en `http://localhost:8501` con 3 pestañas:

1. **Etiquetar entrenamiento** — selecciona la carpeta con tus imágenes
   (botón nativo del sistema operativo o escribiendo la ruta) y las vas
   etiquetando con botones en pantalla.
2. **Entrenar modelo** — un botón entrena el modelo, con la validación
   cruzada actualizándose en vivo y, al final, accuracy, precision,
   recall, F1 y matriz de confusión.
3. **Predecir y evaluar** — selecciona la carpeta con las imágenes
   externas (por ejemplo las del profesor), opcionalmente etiqueta el
   ground truth real ahí mismo, y observa las métricas y la matriz de
   confusión actualizándose en vivo mientras predice.

### Si algo falla en Windows

- **`streamlit no se reconoce`** → no está instalado, o hay un problema
  de permisos al instalar. Usa un entorno virtual:
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  python -m streamlit run app.py
  ```
- **La página carga en blanco varios segundos** → normal, TensorFlow
  tarda en cargar la primera vez.
- **No aparecen imágenes al seleccionar carpeta** → revisa que la ruta
  no tenga comillas pegadas (pasa si usaste "Copiar como ruta de acceso"
  en Windows), que las imágenes estén directamente dentro de la carpeta
  (no en una subcarpeta), y usa el cuadro de diagnóstico que aparece en
  la app para confirmar cuántos archivos ve.

---

## 📏 Criterios de etiquetado: ¿qué cuenta como "barco" y qué no?

Usa **exactamente los mismos criterios** al etiquetar las imágenes de
entrenamiento y las de evaluación (ground truth), o la comparación deja
de ser válida.

**Sí cuenta como barco:**
- Cualquier embarcación visible (carguero, pesquero, yate, ferry, etc.),
  navegando o atracada.
- Aunque sea pequeña, si se distingue una silueta de casco reconocible.
- Aunque esté parcialmente cortada por el borde de la imagen.
- Si hay varios barcos, la imagen sigue siendo una sola etiqueta = 1.

**No cuenta como barco:**
- Solo la estela de un barco sin casco visible.
- Muelles o puertos vacíos.
- Boyas, plataformas petroleras, rompeolas.
- Sombras, nubes o formaciones costeras que parecen barcos.

**Casos ambiguos/borrosos:** si no puedes distinguir con confianza,
etiqueta como "no barco" por defecto, y anótalo aparte para poder
explicar esos casos límite.

---

## 📁 Estructura del proyecto

```
barco_ia/
├── app.py                              # App principal (Streamlit)
├── requirements.txt
├── README.md                           # Este archivo
├── labels_train.csv                    # Se genera al etiquetar (Paso 1)
├── ground_truth_predict.csv            # Se genera al etiquetar el ground truth (Paso 3)
├── modelo_barcos.joblib                # Se genera al entrenar (Paso 2)
├── config_app.json                     # Guarda las rutas de carpeta elegidas
├── 1_etiquetar_imagenes.py             # Versión por consola (alternativa a la app)
├── 2_entrenar_modelo.py
├── 3_predecir_imagenes_nuevas.py
├── 4_etiquetar_ground_truth_profesor.py
└── 5_evaluar_resultados_profesor.py
```

## 📦 Requisitos

```
tensorflow>=2.12
scikit-learn>=1.2
matplotlib>=3.7
numpy>=1.23
pillow>=9.4
joblib>=1.2
streamlit>=1.30
pandas>=1.5
```
