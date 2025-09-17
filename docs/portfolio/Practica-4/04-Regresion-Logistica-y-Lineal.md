---

title: "Práctica 4 — Regresión Lineal"
date: 2025-09-16
----------------

# Práctica 4 — Regresión Lineal y Logística 

## Contexto

- En esta práctica exploramos dos enfoques fundamentales de *Machine Learning supervisado*:  
1. **Regresión lineal** aplicada al dataset **Boston Housing**, para predecir valores continuos (precio de casas).  
2. **Regresión logística** aplicada al dataset **Breast Cancer**, para clasificar tumores como **benignos o malignos**.  
- El objetivo fue no solo aplicar modelos, sino también comprender sus diferencias, interpretar métricas y reflexionar sobre el valor práctico que aportan en distintos contextos: económico y médico.  

## Marco Teórico

**Machine Learning (ML)** es la disciplina que enseña a las computadoras a aprender patrones a partir de datos para realizar predicciones. Su funcionamiento se inspira en cómo aprende el cerebro humano: observar muchos ejemplos, detectar patrones y luego aplicarlos a casos nuevos.

### Tipos de aprendizaje
- **Supervisado**: el modelo aprende a partir de ejemplos con etiquetas (ej.: predecir el precio de una casa, clasificar un tumor como benigno o maligno).  
- **No supervisado**: el modelo busca patrones ocultos en datos sin etiquetas (ej.: segmentación de clientes).  
- **Semi-supervisado**: combinación de ambos enfoques, útil cuando hay pocas etiquetas.  
- **Reinforcement learning**: el modelo (agente) aprende a partir de prueba y error en un entorno.

### Proceso CRISP-DM
El ciclo típico de un proyecto de ML sigue estos pasos:
1. **Comprensión del negocio**  
2. **Comprensión de los datos**  
3. **Preparación de datos**  
4. **Modelado**  
5. **Evaluación**  
6. **Despliegue**  

### Conceptos clave
- **Train/Test Split**: separar los datos en entrenamiento y prueba permite evaluar la capacidad de generalización del modelo.  
- **Métricas en regresión**: MAE, MSE, RMSE, R².  
- **Métricas en clasificación**: Accuracy, Precision, Recall, F1-score y matriz de confusión.  

### Modelos utilizados
- **Regresión lineal**: predice valores continuos a partir de una relación lineal entre variables.  
- **Regresión logística**: modelo lineal que usa función sigmoide para generar probabilidades y clasificar en categorías binarias.  


## Objetivos

- Aprender a **cargar y explorar datasets reales**.  
- Diferenciar entre problemas de **regresión (números continuos)** y **clasificación (categorías)**.  
- Entrenar modelos básicos de **regresión lineal** y **logística** con *Scikit-learn*.  
- Evaluar modelos usando métricas apropiadas en cada caso.  
- Entender la importancia de separar datos en **entrenamiento y prueba** para evitar el sobreajuste.  
- Relacionar los resultados con **interpretaciones prácticas** en contextos reales.  

## Actividades (con tiempos estimados)

| Actividad                         | Tiempo | Resultado esperado                                  |
| --------------------------------- | :----: | --------------------------------------------------- |
| Setup inicial                     |   5m   | Librerías cargadas y entorno listo.                 |
| Cargar Dataset de Boston Housing  |   5m   | X con 13 features, y con precios                    |
| Entrenamiento de regresión lineal |   20m  | Modelo entrenado y predicciones                     |
| Bonus                             |   10m  |  Definiciones completas                             |
| Cargar datos médicos              |   5m   |Dataset con 569 pacientes y 30 características listo |
| Entrenar regresión logística      |   20m  | Modelo entrenado y predicciones                     |
| Bonus                             |   5m   |  Preguntas contestadas                              |
| Preguntas de reflexión            |   15m  |   Reflexionar sobre lo hecho en la práctica         |
| Comparación simple                |   10m  | Comparar las dos regresiones                        |
| Reflexión final                   |   15m  |  Conclusiones sobre la práctica                     |

## Desarrollo

### 🏠 1. Carga del dataset

 - 🔧 Se utilizó el dataset **Boston Housing** disponible en `sklearn.datasets`. Contiene 506 observaciones y 13 variables predictoras como: `CRIM` (índice de criminalidad), `ZN` (proporción de terrenos residenciales), `NOX` (concentración de óxidos de nitrógeno), entre otras. La variable objetivo es `MEDV`, que indica el valor medio de las casas.

#### 💡 PISTAS:
- **LinearRegression**se usa para el modelo de regresión lineal (según documentación de sklearn).
- **train_test_split** permite dividir los datos en entrenamiento y prueba.
- Las métricas (**mean_squared_error, mean_absolute_error, r2_score**) son las adecuadas para evaluar modelos de regresión.

### 🏠 2. División en entrenamiento y prueba

#### 🤖 Explicación de blanks:
- Se elimina medv de X porque es la variable que queremos predecir.
- y = boston_data['medv'] toma únicamente la columna de precios.

#### ✅ Resultado de ejecución:
- Dataset con 506 filas y 14 columnas.
- X → 13 variables independientes.
- y → vector con los precios (medv).
- Rango de precios: $5k – $50k.

### 🏠 3. Entrenamiento del modelo

- 🔬 Se entrenó un modelo de **regresión lineal** con `LinearRegression` de Scikit-learn. El modelo estima una relación lineal entre las variables predictoras y el precio medio de las viviendas.

#### 🤖 Explicación de blanks:
- **LinearRegression()** crea el modelo.
- **.fit(X_train, y_train)** entrena el modelo usando datos de entrenamiento.
- **.predict(X_test)** genera predicciones sobre datos no vistos.

#### ✅ Resultado de ejecución:
- Entrenamiento sobre 404 casas.
- Prueba sobre 102 casas.
- Predicciones generadas exitosamente.

- 🔍 Se aplicó el modelo entrenado sobre los datos de prueba, obteniendo predicciones de `MEDV` que luego fueron comparadas con los valores reales.

#### Métricas de evaluación:
- **MAE:** $3.19k
- **MSE:** 24.29
- **RMSE:** $4.93k
- **R²:** 0.669
- **MAPE:** 16.9%

#### Interpretación de resultados

* El **R² = 0.73** indica que el 66.9% de la variabilidad del precio medio de las casas puede explicarse por las variables incluidas en el modelo.
* Un **RMSE ≈ 4.93** significa que, en promedio, las predicciones del modelo tienen un error de ±5 mil dólares respecto al valor real de las viviendas.
* El **MAE = 3.19** refuerza la idea de que el modelo se aproxima bastante a los valores reales, aunque con cierto margen de error.
* El modelo es relativamente bueno, pero no perfecto. Factores no capturados en el dataset (ejemplo: dinámica económica, ubicación exacta de las casas) explican el error residual.

### 📚 BONUS

📈 Para evaluar el rendimiento se calcularon las siguientes métricas:

* **MSE (Mean Squared Error)**: Promedio de los errores al cuadrado, penaliza más los errores grandes.
* **RMSE (Root Mean Squared Error)**: Raíz cuadrada del MSE, vuelve a las unidades originales del problema.
* **MAE (Mean Absolute Error)**: Promedio de los errores absolutos sin importar si son positivos o negativos.
* **R² (Coeficiente de determinación)**: Indica qué porcentaje de la variabilidad es explicada por el modelo (0-1, donde 1 es perfecto).
* **MAPE**: Error porcentual promedio, útil para comparar modelos con diferentes escalas.

### 🏠 4. Cargar datos médicos 
- 📋 Contexto de negocio (CRISP-DM: Business Understanding)
- Problema: Un hospital necesita asistencia automatizada para diagnóstico de cáncer de mama.
- Objetivo: Clasificar tumores como benignos (1) o malignos (0) a partir de características celulares.
- Variables: 30 características de núcleos celulares (ej: radio, textura, perímetro, área, suavidad, etc.).
- Valor para el negocio: Proveer soporte a médicos reduciendo tiempo de análisis, aumentando precisión y sirviendo como segunda opinión automática.
#### ✅ Resultado de ejecución:
- Tumores malignos: 212 casos (≈ 37%)
- Tumores benignos: 357 casos (≈ 63%)
- Esta distribución muestra que el dataset no está perfectamente balanceado, aunque la diferencia no es extrema. Aun así, es importante tenerlo en cuenta porque podría influir en el rendimiento de los clasificadores, especialmente en métricas como precisión, recall y F1-score.

### 🏠 5. Entrenar regresión logística:

#### 💡 PISTAS:
- **train_test_split** se utiliza para dividir el dataset en entrenamiento (80%) y prueba (20%), garantizando aleatoriedad con random_state=42.
- **LogisticRegression** es la clase de sklearn.linear_model para entrenar un modelo de regresión logística. El parámetro max_iter=5000 asegura que el algoritmo tenga suficiente número de iteraciones para converger.
- Los métodos **.fit()** y **.predict()** funcionan igual que en regresión lineal.
- Las métricas usadas son:
* accuracy_score: proporción de predicciones correctas.
* precision_score: proporción de verdaderos positivos sobre todos los positivos predichos.
* recall_score: proporción de verdaderos positivos sobre todos los positivos reales.
* f1_score: media armónica entre precisión y recall.
- **confusion_matrix** permite ver los aciertos y errores en términos de verdaderos/falsos positivos y negativos.
- **classification_report** genera un desglose detallado por clase.

#### ✅ Resultado de ejecución:

- Entrenamiento: 455 pacientes
- Prueba: 114 pacientes

- Métricas de clasificación:
- Accuracy: 95.6%
- Precision: 94.6%
- Recall: 98.6%
- F1-Score: 0.966

- Verdaderos Negativos: 39
- Falsos Positivos: 4

- Falsos Negativos: 1
- Verdaderos Positivos: 70

#### Interpretación de resultados

- El modelo tiene una alta exactitud (95.6%), lo que indica que clasifica correctamente la mayoría de los casos.

- Precisión (94.6%): de los tumores predichos como benignos, el 94.6% lo eran realmente.

- Recall (98.6%): de todos los tumores benignos reales, el modelo identificó casi todos.

- F1-Score (0.966): muestra un excelente balance entre precisión y recall.

- La matriz de confusión indica que solo hubo 5 errores en 114 predicciones (4 falsos positivos y 1 falso negativo).

- Desde un punto de vista médico, el bajo número de falsos negativos (solo 1) es crucial, ya que significa que casi no se dejan pasar casos malignos como benignos, reduciendo el riesgo para los pacientes.

### 🎁  BONUS: ¿Qué significan las métricas de clasificación?

- **Accuracy:** Porcentaje de predicciones **correctas** sobre el total.  
- **Precision:** De todas las predicciones **positivas**, ¿cuántas fueron realmente correctas? (94,6%)
- **Recall (Sensibilidad):** De todos los casos **positivos reales**, ¿cuántos detectamos? (98,6%)
- **F1-Score:** Promedio **armónico** entre precision y recall.  
- **Matriz de Confusión:** Tabla que muestra **valores reales** vs **valores predichos**.  

### 🧠 Paso 6: Preguntas de Reflexión  

**1. ¿Cuál es la diferencia principal entre regresión lineal y logística?**  
- La **regresión lineal** predice valores **continuos** (por ejemplo, el precio de una casa).  
- La **regresión logística** predice valores **categóricos/binarios** (por ejemplo, benigno vs maligno).  

**2. ¿Por qué dividimos los datos en entrenamiento y prueba?**  
- Para asegurarnos de que el modelo no solo memorice los datos, sino que también pueda **generalizar a datos nuevos**.  
- El **train/test split** permite entrenar con una parte de los datos y luego evaluar el desempeño en ejemplos que el modelo nunca vio.  


**3. ¿Qué significa una exactitud del 95%?**  
- Significa que de cada **100 pacientes**, el modelo clasifica correctamente a **95**.  
- En nuestro caso, de 114 pacientes de prueba, el modelo acertó en 109 y se equivocó en 5.  


**4. ¿Cuál es más peligroso: predecir "benigno" cuando es "maligno", o al revés?**  
- Es **más peligroso predecir "benigno" cuando en realidad es maligno**, porque el paciente podría no recibir tratamiento a tiempo.  
- En cambio, predecir "maligno" cuando era benigno genera preocupación innecesaria, pero no pone en riesgo la vida del paciente.  

### 🔍 Paso 7: Comparación Simple  

| Aspecto            | Regresión Lineal                         | Regresión Logística                          |
| ------------------ | ---------------------------------------- | -------------------------------------------- |
| Qué predice        | Valores numéricos continuos              | Categorías (clases, ej. 0 o 1)               |
| Ejemplo de uso     | Precio de una casa en dólares            | Diagnóstico de tumor: benigno/maligno        |
| Rango de salida    | Números reales (-∞, +∞)                  | Probabilidades entre 0 y 1 → luego clase 0/1 |
| Métrica principal  | Error (MAE, MSE, RMSE, R², MAPE)         | Exactitud, Precisión, Recall, F1-Score       |

### 📝 Paso 8: Reflexión Final  

**1. ¿Cuál modelo usarías para predecir el salario de un empleado?**  
- Usaría **regresión lineal**, porque el salario es un **valor continuo** que puede tomar muchos posibles montos.  

**2. ¿Cuál modelo usarías para predecir si un email es spam?**  
- Usaría **regresión logística**, porque el problema es de **clasificación binaria** (spam o no spam).  

**3. ¿Por qué es importante separar datos de entrenamiento y prueba?**  
- Porque permite comprobar si el modelo **generaliza bien a datos nuevos**.  
- Si solo usamos datos de entrenamiento, el modelo puede “memorizar” (overfitting).  
- Con datos de prueba podemos evaluar el **desempeño real** en situaciones que no vio durante el entrenamiento.  


## Evidencias

* [Código completo para ejecutar en Google Colab](https://colab.research.google.com/drive/14INyAU9dGAbxu2TPs-GuP_bgt4n8zkKS?usp=sharing)

### Código con los espacios en blanco rellenados que se ejecutó:

```python
# 1. Setup inicial:
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

# 2. Cargar dataset
X = boston_data.drop('medv', axis=1)  # Todas las columnas EXCEPTO la que queremos predecir
y = boston_data['medv']                # Solo la columna que queremos predecir

# 3. Entrenar modelo
modelo_regresion = LinearRegression()
modelo_regresion.fit(X_train, y_train)

predicciones = modelo_regresion.predict(X_test)

mae = mean_absolute_error(y_test, predicciones)
mse = mean_squared_error(y_test, predicciones)
r2 = r2_score(y_test, predicciones)

# 6. Entrenar regresión logística
X_train_cancer, X_test_cancer, y_train_cancer, y_test_cancer = train_test_split( X_cancer, y_cancer, test_size=0.2, random_state=42
)

modelo_clasificacion = LogisticRegression(max_iter=5000, random_state=42)
modelo_clasificacion.fit(X_train_cancer, y_train_cancer)

predicciones_cancer = modelo_clasificacion.predict(X_test_cancer)

precision = precision_score(y_test_cancer, predicciones_cancer)
recall = recall_score(y_test_cancer, predicciones_cancer)
f1 = f1_score(y_test_cancer, predicciones_cancer)

matriz_confusion = confusion_matrix(y_test_cancer, predicciones_cancer)

print(classification_report(y_test_cancer, predicciones_cancer, target_names=['Maligno', 'Benigno']))

   
```


## Reflexión
A lo largo de la práctica se aprendió:  

- **Cómo cargar y explorar datos reales** de distintas áreas (economía y salud).  
- La **diferencia clave entre regresión lineal y logística**: valores continuos vs categorías.  
- A entrenar modelos de *machine learning* desde cero con pasos básicos (`train_test_split`, `.fit()`, `.predict()`).  
- La importancia de **elegir las métricas adecuadas** según el tipo de problema (MAE, RMSE, R² en regresión; accuracy, precision, recall y F1 en clasificación).  
- El valor de separar datos en **entrenamiento y prueba** para evaluar el rendimiento real de un modelo.  

- En conclusión, la práctica permitió desarrollar una visión clara de cómo aplicar modelos sencillos de ML y, sobre todo, **interpretar sus resultados en contextos reales**: desde estimar precios de viviendas hasta apoyar diagnósticos médicos.  

---
## Referencias

- [Scikit-learn — Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)  
- [Scikit-learn — Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)  
- [Métricas de evaluación en Scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html)  
- Harrison, D. & Rubinfeld, D.L. (1978). *Hedonic prices and the demand for clean air*. Journal of Environmental Economics and Management, 5(1), 81–102. (Dataset original Boston Housing)  
- [UCI Machine Learning Repository — Breast Cancer Wisconsin Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)  