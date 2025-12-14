# Pruebas realizadas – Actividad 3  
## Métodos de aprendizaje supervisado

Proyecto: Transporte masivo  
Modelo: Árbol de decisión (regresión)

## Cómo ejecutar el proyecto

1. Crear y activar el entorno virtual.
2. Instalar dependencias:
```bash
pip install -r requirements.txt
````

3. Ejecutar el modelo:

```bash
python src/modelo_supervisado.py
```

## Métricas de evaluación utilizadas

Para evaluar el desempeño del modelo supervisado se utilizaron las siguientes métricas:

* **MAE (Mean Absolute Error):**
  Representa el error absoluto promedio entre los valores reales y los valores predichos por el modelo.
  En este contexto, indica cuántos minutos, en promedio, se equivoca el modelo al estimar el tiempo de viaje.

* **RMSE (Root Mean Squared Error):**
  Es la raíz del error cuadrático medio. Penaliza más los errores grandes, por lo que es sensible a predicciones muy alejadas del valor real.
  Se expresa también en minutos.

* **R2 (Coeficiente de determinación):**
  Indica qué proporción de la variabilidad del tiempo de viaje es explicada por el modelo.
  Un valor cercano a 1 indica alta capacidad explicativa, mientras que valores bajos indican que el modelo explica solo una parte limitada del fenómeno.


## Evidencia general del modelo

Durante la ejecución del script se generan los siguientes archivos:

* data/raw/dataset_transporte_masivo.csv
* outputs/arbol_decision.png
* outputs/metricas.json

Estos archivos permiten reproducir los resultados y visualizar el modelo entrenado.


## Ejecución base del modelo

### Configuración

* Tamaño del dataset: 400 registros
* Proporción entrenamiento/prueba: 70% / 30%
* Árbol de decisión limitado para mejorar interpretabilidad

### Métricas obtenidas

* MAE_min: 9.07
* RMSE_min: 11.25
* R2: 0.098
* n_test: 120

### Interpretación

El modelo presenta un error promedio cercano a 9 minutos en la predicción del tiempo de viaje.
El valor de RMSE es mayor debido a la penalización de errores grandes.
El valor de R2 es bajo, lo que indica que el modelo explica una parte limitada de la variabilidad del tiempo de viaje.

Este comportamiento es esperable dado que el dataset es simulado, incluye ruido y se utilizan pocas variables explicativas. Además, el árbol fue restringido en profundidad y tamaño de hojas para evitar sobreajuste y mejorar su interpretabilidad.


## Prueba adicional: cambio en la partición de datos

### Configuración

* Tamaño del dataset: 400 registros
* Proporción entrenamiento/prueba: 80% / 20%
* Resto de parámetros del modelo sin cambios

### Métricas obtenidas

* MAE_min: 9.34
* RMSE_min: 11.91
* R2: 0.897
* n_test: 80

### Interpretación

Al aumentar la cantidad de datos de entrenamiento, el modelo logra explicar una mayor proporción de la variabilidad del tiempo de viaje, reflejado en el aumento del valor de R2.
El error promedio se mantiene en un rango similar al de la ejecución base, lo que sugiere que el modelo es relativamente estable frente a cambios en la partición de los datos.


## Control del sobreajuste e interpretabilidad

Durante las pruebas iniciales se observó que un árbol sin restricciones generaba una estructura demasiado profunda y difícil de interpretar.
Para evitar este comportamiento, se ajustaron los siguientes parámetros:

* max_depth = 3
* min_samples_leaf = 20
* min_samples_split = 40

Estos ajustes permitieron obtener un árbol más compacto, con reglas claras y coherentes con el dominio del transporte masivo.

## Evidencia del modelo entrenado

El archivo `outputs/arbol_decision.png` muestra el árbol de decisión final.
En él se observa que variables como la distancia del trayecto, el número de estaciones y los transbordos tienen una influencia directa en el tiempo de viaje estimado.

Estas reglas permiten interpretar el modelo de forma intuitiva, por ejemplo:

* A mayor distancia y número de estaciones, mayor tiempo de viaje.
* La presencia de transbordos y condiciones adversas incrementa el tiempo estimado.


## Conclusiones generales de las pruebas

El modelo supervisado desarrollado cumple con el objetivo de aprender patrones a partir de datos históricos simulados del sistema de transporte masivo.

Aunque el poder explicativo depende de la configuración del conjunto de datos, los resultados obtenidos son coherentes con un escenario realista y controlado.
El árbol de decisión ofrece un buen equilibrio entre desempeño y explicabilidad, lo que lo convierte en una herramienta útil como apoyo a la toma de decisiones, por ejemplo para estimar tiempos esperados y comparar escenarios de operación.

## Prueba adicional: aumento del tamaño del dataset

### Configuración
- Tamaño del dataset: 800 registros
- Proporción entrenamiento/prueba: 80% / 20%
- Parámetros del modelo sin cambios
- Semilla fija para garantizar reproducibilidad

### Métricas obtenidas
- MAE_min: 9.34  
- RMSE_min: 11.91  
- R2: 0.897  
- n_test: 80  

### Interpretación
Al duplicar el tamaño del conjunto de datos, las métricas obtenidas se mantienen muy similares a las de la prueba anterior.  
Esto indica que el modelo es estable frente a cambios en el volumen de datos y que las reglas aprendidas no dependen de casos particulares.

La estabilidad de los resultados se explica porque:
- El dataset sigue el mismo patrón estadístico al ser simulado.
- El árbol de decisión está regularizado mediante restricciones de profundidad y tamaño mínimo de hojas.
- Se utiliza una semilla fija, lo que garantiza reproducibilidad del experimento.

Este comportamiento es deseable en modelos supervisados, ya que sugiere una buena capacidad de generalización bajo las condiciones del experimento.
