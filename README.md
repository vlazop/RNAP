# 🚗 Sistema de Clasificación y Detección de Defectos en Neumáticos

Este proyecto desarrolla una solución basada en **Deep Learning** y **Visión por Computadora** para la clasificación y detección automática de defectos en las texturas de neumáticos. Está diseñado para abordar problemas reales en el ámbito de la inspección industrial y la seguridad vial utilizando redes neuronales convolucionales.

El desarrollo se encuentra implementado paso a paso en el notebook de Jupyter [Preg_01.ipynb](file:///c:/Users/vlazo/workspace/MIA/RNAP-1/Preg_01.ipynb).

<a href="https://colab.research.google.com/github/vlazop/RNAP/blob/main/Preg_01.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

---

## 📋 Índice
1. [Descripción del Proyecto](#descripción-del-proyecto)
2. [Dataset](#dataset)
3. [Tecnologías y Librerías](#tecnologías-y-librerías)
4. [Estructura del Proyecto y Metodología](#estructura-del-proyecto-y-metodología)
   - [1. Configuración e Ingesta de Datos](#1-configuración-e-ingesta-de-datos)
   - [2. Tratamiento de Desbalance (WeightedRandomSampler)](#2-tratamiento-de-desbalance-weightedrandomsampler)
   - [3. Arquitecturas Comparadas](#3-arquitecturas-comparadas)
   - [4. Bucle de Entrenamiento y Focal Loss](#4-bucle-de-entrenamiento-y-focal-loss)
   - [5. Evaluación del Modelo](#5-evaluación-del-modelo)
   - [6. Interpretabilidad con Grad-CAM](#6-interpretabilidad-con-grad-cam)
   - [7. Ajuste de Hiperparámetros](#7-ajuste-de-hiperparámetros)
   - [8. Análisis Cualitativo de Fallos](#8-análisis-cualitativo-de-fallos)
5. [Resultados y Conclusiones](#resultados-y-conclusiones)
6. [Cómo Ejecutar el Proyecto](#cómo-ejecutar-el-proyecto)

---

## 🔍 Descripción del Proyecto

El objetivo principal es clasificar imágenes de texturas de neumáticos en dos categorías bien definidas:
*   **Normal / Sano (Clase 0):** Neumáticos en buen estado sin grietas ni daños significativos.
*   **Dañado / Quebrado (Clase 1):** Neumáticos con defectos superficiales, desgaste irregular o roturas estructurales.

Para lograr esto, se realiza un estudio comparativo (estudio de ablación) entre una red convolucional personalizada (`CustomCNN`) entrenada desde cero y un modelo de transferencia de aprendizaje (`ResNet-50`) preentrenado con ImageNet.

---

## 📊 Dataset

Se hace uso del conjunto de datos **[Tire Texture Image Recognition](https://www.kaggle.com/datasets/jehanbhathena/tire-texture-image-recognition)** disponible en Kaggle, el cual cuenta con imágenes agrupadas en:
*   `training_data`: Imágenes de entrenamiento para las clases normal y defectuosa.
*   `testing_data`: Imágenes para la validación y testeo.

---

## 🛠️ Tecnologías y Librerías

El ecosistema tecnológico utilizado incluye:
*   **Lenguaje:** Python 3
*   **Framework de Deep Learning:** PyTorch (`torch`, `torchvision`)
*   **Métricas de Evaluación:** Scikit-Learn (`sklearn`)
*   **Interpretabilidad:** `pytorch-grad-cam` (Grad-CAM)
*   **Visualización de datos:** Matplotlib, Seaborn

---

## 📂 Estructura del Proyecto y Metodología

### 1. Configuración e Ingesta de Datos
El entorno requiere configurar el token de Kaggle (`kaggle.json`) para descargar y descomprimir el dataset en el directorio local:
```bash
# Descarga automática en el entorno de ejecución
kaggle datasets download -d jehanbhathena/tire-texture-image-recognition
unzip -q tire-texture-image-recognition.zip -d tire_data/
```

### 2. Tratamiento de Desbalance (WeightedRandomSampler)
Debido a la disparidad en la cantidad de muestras de cada clase, se implementó una estrategia de balanceo por lotes:
*   **Aumentación de datos:** Redimensionado a 224x224 píxeles, volteo horizontal aleatorio y normalización basada en estadísticas de ImageNet.
*   **Samplers ponderados:** `WeightedRandomSampler` asigna una probabilidad inversa a la frecuencia de la clase para cada muestra, garantizando que el modelo sea expuesto equitativamente a ambas clases en cada época.

### 3. Arquitecturas Comparadas

#### A. Custom CNN (Desde Cero)
Consta de 3 bloques convolucionales secuenciales y una sección completamente conectada:
*   **Capas:** 3 × (Conv2d -> ReLU -> MaxPool2d) incrementando canales de 3 -> 32 -> 64 -> 128.
*   **Clasificador:** Capa densa con `Dropout(0.5)` para evitar sobreajuste, finalizando en un solo nodo con activación lineal (compatible con pérdidas con logits).

#### B. ResNet-50 (Transfer Learning)
Se adaptó una arquitectura ResNet-50 preentrenada en ImageNet:
*   Se congelaron los pesos base de extracción de características.
*   Se reemplazó la capa totalmente conectada terminal por un clasificador de 2 capas: `Linear(2048, 256) -> ReLU -> Dropout(0.3) -> Linear(256, 1)`.

### 4. Bucle de Entrenamiento y Focal Loss
En el entrenamiento de problemas desbalanceados, los ejemplos fáciles de clasificar pueden inundar el gradiente. Para mitigar esto, además de la pérdida estándar de entropía cruzada binaria (BCE), se implementó **Focal Loss**:

`FL(pt) = -αt * (1 - pt)^γ * log(pt)`

Donde se configuran los hiperparámetros α = 1 y el factor de focalización γ = 2 para enfocar el gradiente en las muestras más difíciles.

![Curvas de convergencia de pérdida](Informe/Imagenes/grafico_perdidas.png)

### 5. Evaluación del Modelo
La validación se realiza de manera exhaustiva utilizando:
*   **Exactitud (Accuracy)**
*   **Puntuación F1 (F1-Score)**
*   **Área Bajo la Curva ROC (AUC-ROC)**
*   **Matriz de Confusión** (implementada con mapas de calor de Seaborn para ilustrar falsos positivos y falsos negativos).

![Comparación de Métricas](Informe/Imagenes/comparacion_metricas.png)

#### Matrices de Confusión comparadas

| Custom CNN (Focal Loss) | ResNet-50 (Focal Loss) | ResNet-50 (BCE) |
| :---: | :---: | :---: |
| ![Custom CNN CM](Informe/Imagenes/matriz_confusion_scratch.png) | ![ResNet-50 FL CM](Informe/Imagenes/matriz_confusion.png) | ![ResNet-50 BCE CM](Informe/Imagenes/matriz_confusion_bce.png) |

### 6. Interpretabilidad con Grad-CAM
Para asegurar que el modelo está aprendiendo los patrones de las grietas y no sesgándose por elementos del fondo de la imagen, se generaron mapas de activación de la última capa convolucional de la ResNet-50 (`model_resnet.layer4[-1]`). 

Esto superpone un mapa térmico (Grad-CAM) sobre la textura del neumático indicando qué partes influyen más en la decisión del modelo.

![Visualización Grad-CAM](Informe/Imagenes/gradcam_resultado.png)

### 7. Ajuste de Hiperparámetros
Se evaluó el comportamiento de la red entrenando con diferentes tasas de aprendizaje:
*   **Learning Rates comparados:** 2e-4 vs 1e-4 usando optimizador Adam (siguiendo la regla de escalado lineal con batch=64).
*   El modelo ResNet-50 demostró un rendimiento superior y estable con un learning rate de 0.0002 (2e-4).

### 8. Análisis Cualitativo de Fallos
Se extrajeron y visualizaron 5 imágenes en las cuales el modelo de Transfer Learning falló en predecir correctamente la etiqueta real, permitiendo formular hipótesis sobre las causas:
1.  **Texturas Ambiguas:** Presencia de suciedad o brillo en el caucho que asemejan grietas.
2.  **Sombras en los Surcos:** Iluminación deficiente interpretada erróneamente como discontinuidad física o corte.
3.  **Frontera de Decisión Incipiente:** Grietas extremadamente pequeñas que el clasificador etiqueta como normales.
4.  **Desenfoque/Falta de Foco:** Pérdida de nitidez espacial que disminuye el contraste en las altas frecuencias de la imagen.

![Muestras de Fallo](Informe/Imagenes/analisis_fallos.png)

---

## 📈 Resultados y Conclusiones

| Métrica | Custom CNN (FL) | ResNet-50 (FL) | ResNet-50 (BCE) |
| :--- | :---: | :---: | :---: |
| **Exactitud (Accuracy)** | 66.77% | 77.85% | **79.69%** |
| **Precision** | 51.89% | 63.19% | **66.01%** |
| **Recall** | 83.48% | **89.57%** | 87.83% |
| **F1-Score** | 64.00% | 74.10% | **75.37%** |
| **AUC-ROC** | 77.13% | **89.28%** | 87.94% |
| **Mejor Pérdida (Train)** | 0.0244 | **0.0100** | 0.0725 |

### Conclusiones Principales:
*   **Transfer Learning es Indispensable:** ResNet-50 demostró una velocidad de convergencia drásticamente mayor y una mejora en exactitud de casi 13 puntos porcentuales en comparación con el modelo personalizado.
*   **Mitigación Efectiva:** La combinación de `WeightedRandomSampler` con la estrategia de pérdidas balanceó de forma robusta el aprendizaje, logrando recalls elevados de hasta **89.57%** en la detección de defectos.
*   **Estudio de Ablación de Pérdida:** Aunque BCE logró una exactitud marginalmente superior (79.69%), Focal Loss demostró un mejor AUC-ROC (89.28%), confirmando una separación de clases probabilística más robusta frente al desequilibrio de datos.
*   **Validación de Criterio (Interpretabilidad):** Mediante Grad-CAM se corroboró que el modelo enfoca su atención principalmente en las fisuras y deformidades texturales de las llantas, descartando atajos contextuales del fondo.

---

## 🚀 Cómo Ejecutar el Proyecto

1.  **Instalar dependencias necesarias:**
    ```bash
    pip install torch torchvision torcheval grad-cam scikit-learn matplotlib seaborn kaggle
    ```
2.  **Configurar credenciales de Kaggle:**
    Descarga tu archivo `kaggle.json` de tu perfil de Kaggle y colócalo en el directorio raíz o en `~/.kaggle/`.
3.  **Abrir y ejecutar el notebook:**
    Puedes ejecutar todas las celdas de [Preg_01.ipynb](file:///c:/Users/vlazo/workspace/MIA/RNAP-1/Preg_01.ipynb) de manera local en tu IDE favorito (como VS Code) o en Google Colab usando el botón de "Open In Colab" superior.