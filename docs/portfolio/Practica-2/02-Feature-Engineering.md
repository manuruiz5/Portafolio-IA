---
title: "Práctica 2 — Feature Engineering simple + Modelo base"
date: 2025-08-16
---

# Práctica 2 — Feature Engineering simple + Modelo base

## Contexto
En esta práctica trabajamos sobre el dataset del Titanic para aplicar técnicas de **Feature Engineering** y entrenar un **modelo base de clasificación**.  

Además, exploramos los componentes principales de Scikit-learn para clasificación y validación de modelos.

> **Nota**: Trabajamos con el dataset **Titanic** de Kaggle, que contiene información de pasajeros y si sobrevivieron o no.  

## Objetivos
- Comprender el uso de `LogisticRegression`, `DummyClassifier`, `train_test_split` y métricas de evaluación en Scikit-learn.  
- Crear nuevas variables a partir de las originales para mejorar el rendimiento del modelo.  
- Entrenar un modelo baseline y compararlo con una regresión logística.  

## Actividades (con tiempos estimados)

| Actividad                              | Tiempo | Resultado esperado                            |
|----------------------------------------|:------:|-----------------------------------------------|
| Investigación de Scikit-learn          | 10m    | Conocer componentes clave                     |
| Preprocesamiento y Feature Engineering | 15m    | Dataset enriquecido y limpio                  |
| Entrenamiento baseline y LogReg        | 20m    | Modelos entrenados y métricas obtenidas       |
| Análisis de resultados                 | 15m    | Reflexiones sobre errores y mejoras           |

## Desarrollo

### 1. Investigación inicial
- **LogisticRegression**: Modelo lineal para clasificación binaria/multiclase. Importante ajustar `solver`, `C`, `max_iter`. `liblinear` funciona bien para datasets pequeños.  
- **DummyClassifier**: Proporciona un baseline simple (ej. siempre predecir la clase más frecuente). Sirve para validar si un modelo “aprende” algo más allá del azar.  
- **train_test_split**: `stratify` asegura proporciones similares entre train/test, `random_state` asegura reproducibilidad.  
- **Métricas de evaluación**:  
  - *Accuracy*: proporción de aciertos.  
  - *Precision/Recall/F1*: más informativas cuando las clases están desbalanceadas.  
  - *Matriz de confusión*: muestra aciertos y errores por clase.  

### 2. Feature Engineering
- Imputación de valores faltantes: `Embarked` (moda), `Fare` (mediana), `Age` (mediana por sexo y clase).  
- Nuevas variables:
  - `FamilySize = SibSp + Parch + 1`  
  - `IsAlone` (indicador binario)  
  - `Title` (extraído de `Name`, agrupando títulos poco frecuentes en “Rare”).  
- Resultado: **(891, 14) features** finales. (Ver código en "Evidencias")

### 3. Modelado
- Baseline (DummyClassifier, estrategia "most_frequent").  
- Logistic Regression (`liblinear`, `max_iter=1000`).  
- División: 80% train, 20% test, estratificada.  
- Resultados (Ver código en "Evidencias"):
    - Baseline acc: 0.6145
    - LogReg acc : 0.8156


### 4. Evaluación
**Classification report (LogReg):**

| Clase             | Precision | Recall | F1-score | Soporte |
|-------------------|-----------|--------|----------|---------|
| 0 (no sobrevivió) |    0.82   |  0.89  |   0.86   |   110   |
| 1 (sobrevivió)    |    0.80   |  0.70  |   0.74   |    69   |
| Accuracy          |           |        |   0.82   |   179   |
| macro avg         |    0.81   |  0.79  |   0.80   |   179   |
| weighted avg      |    0.81   |  0.82  |   0.81   |   179   |  

**Matriz de confusión (LogReg):**

|  Real          | Pred. No sobrevivió | Pred. Sobrevivió |
|----------------|---------------------|------------------|
|  No sobrevivió |         98          |         12       |
|  Sobrevivió    |         21          |         48       |

## Evidencias
- [Notebook en Google Colab](https://colab.research.google.com/drive/1pCw_QCZsqQB8gcLmOhqY-XBz3gmItlxT?usp=sharing).
- **Código ejecutado para la parte 2. Feature Engineering:**
```python
df = train.copy()

# 🚫 PASO 1: Manejar valores faltantes (imputación)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])  # Valor más común
df['Fare'] = df['Fare'].fillna(df['Fare'].median())              # Mediana
df['Age'] = df['Age'].fillna(df.groupby(['Sex','Pclass'])['Age'].transform('median'))

# 🆕 PASO 2: Crear nuevas features útiles
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

df['Title'] = df['Name'].str.extract(',\s*([^\.]+)\.')
rare_titles = df['Title'].value_counts()[df['Title'].value_counts() < 10].index
df['Title'] = df['Title'].replace(rare_titles, 'Rare')

# 🔄 PASO 3: Preparar datos para el modelo
features = ['Pclass','Sex','Age','Fare','Embarked','FamilySize','IsAlone','Title','SibSp','Parch']
X = pd.get_dummies(df[features], drop_first=True)
y = df['Survived']

X.shape, y.shape
```

- **Código ejecutado para la parte 3.Modelado:**
```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

dummy = DummyClassifier(strategy='most_frequent', random_state=42)
dummy.fit(X_train, y_train)
baseline_pred = dummy.predict(X_test)

lr = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)
lr.fit(X_train, y_train)
pred = lr.predict(X_test)

print('Baseline acc:', accuracy_score(y_test, baseline_pred))
print('LogReg acc  :', accuracy_score(y_test, pred))

print('\nClassification report (LogReg):')
print(classification_report(y_test, pred))

print('\nConfusion matrix (LogReg):')
print(confusion_matrix(y_test, pred))
```

## Reflexión
- El modelo se equivoca más al predecir que una persona sobrevivió cuando en realidad no lo hizo (21 errores vs 12 en la otra dirección).  
- Tiene mayor acierto en la clase "no sobrevivió" (recall 0.89 vs 0.70 en "sobrevivió").  
- Comparado con el baseline (61%), la Regresión Logística mejora significativamente (82%).  
- Error más grave: en este contexto, predecir que alguien no sobrevivió cuando en realidad sí lo hizo podría ser más costoso desde el punto de vista humanitario.  
- Mejoras simples: incluir features como agrupar `Age` en categorías (niños, adultos, ancianos).  

## Referencias
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)  
- [Kaggle — Titanic Competition](https://www.kaggle.com/competitions/titanic/data)

