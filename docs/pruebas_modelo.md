# Pruebas realizadas - Modelo supervisado

## Como ejecutar
1. Crear y activar entorno virtual
2. Instalar dependencias: `pip install -r requirements.txt`
3. Ejecutar: `python src/modelo_supervisado.py`

## Evidencia
- Se genera el dataset: `data/raw/dataset_transporte_masivo.csv`
- Se genera el arbol: `outputs/arbol_decision.png`

## Resultados
En consola se imprimen las metricas:
- MAE (min)
- RMSE (min)
- R2

Nota: Los valores pueden variar ligeramente si cambia la semilla o el tamano de la muestra.
