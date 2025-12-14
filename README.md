# Actividad 3 – Métodos de aprendizaje supervisado  
## Inteligencia Artificial

**Proyecto:** Transporte masivo  
**Modelo:** Aprendizaje supervisado (Árbol de decisión – regresión)


### Información académica

- **Estudiante:** Brandon Motta  
- **Programa:** Ingeniería en Ciencia de Datos  
- **Facultad:** Facultad de Ingeniería  
- **Universidad:** Fundación Universitaria Iberoamericana  
- **Asignatura:** Inteligencia Artificial  
- **Actividad:** Actividad 3 – Métodos de aprendizaje supervisado  


## Descripción general del proyecto

Este proyecto hace parte de la **Actividad 3 del curso de Inteligencia Artificial**, cuyo objetivo es aplicar de manera práctica los **métodos de aprendizaje supervisado**.

El contexto del trabajo es un **sistema de transporte masivo**, el mismo dominio utilizado en actividades anteriores del curso.  
Mientras que en actividades previas se abordó el problema desde un enfoque simbólico y de búsqueda de rutas, en esta actividad se aborda el mismo dominio desde un enfoque **basado en datos**, utilizando aprendizaje automático supervisado.

El modelo desarrollado busca **predecir el tiempo de viaje** a partir de variables relacionadas con el trayecto y las condiciones del sistema, tales como distancia, número de estaciones, transbordos, hora pico, clima e incidentes.


## Objetivo del modelo

Construir un modelo de aprendizaje supervisado que:

- Aprenda patrones a partir de datos históricos simulados del sistema de transporte.
- Permita estimar el tiempo de viaje bajo distintas condiciones.
- Sea interpretable, utilizando un árbol de decisión con restricciones para evitar sobreajuste.

Este tipo de modelo puede servir como apoyo para la toma de decisiones, por ejemplo para comparar escenarios de operación o estimar tiempos esperados de viaje.


## Flujo de trabajo implementado

El desarrollo del proyecto sigue el flujo propuesto en clase:

1. Datos  
2. Definición del problema  
3. Preprocesamiento  
4. División del conjunto de datos (train / test)  
5. Selección del modelo supervisado  
6. Entrenamiento del modelo  
7. Evaluación mediante métricas  
8. Análisis de resultados y conclusiones  


## Tecnologías utilizadas

- **Lenguaje:** Python  
- **Versión de Python:** **3.8.10** (requerida para compatibilidad con scikit-learn)  
- **Librerías principales:**
  - numpy
  - pandas
  - scikit-learn
  - matplotlib


## Estructura del repositorio

```

ia-transporte-supervisado/
│
├── src/
│   └── modelo_supervisado.py
│
├── data/
│   └── raw/
│       └── dataset_transporte_masivo.csv
│
├── outputs/
│   ├── arbol_decision.png
│   └── metricas.json
│
├── docs/
│   ├── descripcion_datos.md
│   └── pruebas_modelo.md
│
├── requirements.txt
├── README.md
└── .gitignore

````

## Cómo ejecutar el proyecto

### Instalar dependencias

Con el entorno virtual activo:

```bash
pip install -r requirements.txt
```


### Ejecutar el modelo

```bash
python src/modelo_supervisado.py
```


## Resultados generados

Al ejecutar el script se generan automáticamente:

* Un dataset simulado:
  `data/raw/dataset_transporte_masivo.csv`

* La visualización del árbol de decisión entrenado:
  `outputs/arbol_decision.png`

* Un archivo con las métricas del modelo:
  `outputs/metricas.json`

Además, las métricas y conclusiones se imprimen por consola para facilitar su análisis.


## Evaluación del modelo

El desempeño del modelo se evalúa utilizando las siguientes métricas:

* **MAE (Mean Absolute Error):** error promedio en minutos.
* **RMSE (Root Mean Squared Error):** penaliza errores grandes.
* **R2 (Coeficiente de determinación):** capacidad explicativa del modelo.

Las pruebas realizadas, junto con su interpretación, se encuentran documentadas en:

```
docs/pruebas_modelo.md
```

## Notas finales

El modelo fue diseñado priorizando la **interpretabilidad** sobre la complejidad, mediante la restricción de la profundidad del árbol y el tamaño mínimo de las hojas.
Esto permite comprender de forma clara cómo las variables del sistema influyen en el tiempo de viaje, alineándose con los objetivos formativos del curso de Inteligencia Artificial.
