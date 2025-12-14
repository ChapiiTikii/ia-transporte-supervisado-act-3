# Descripcion de datos - Transporte masivo (Actividad 3)

## Fuente de datos
Para esta actividad se construyo un dataset simulado, basado en variables que normalmente afectan el tiempo de viaje en un sistema de transporte masivo (distancia, numero de estaciones, transbordos, hora pico, clima e incidentes).

Se simulo porque no se dispone de una fuente publica directa con este nivel de detalle para el proyecto.

## Variables
- dist_km: distancia aproximada del trayecto (km)
- num_estaciones: numero de estaciones recorridas
- num_transbordos: cantidad de transbordos realizados
- hora_pico: 1 si es hora pico, 0 si no
- clima_lluvia: 1 si hay lluvia, 0 si no
- incidente: 1 si ocurre un incidente, 0 si no
- dia_semana: 0-6 (dia de la semana)
- tiempo_min (objetivo): tiempo total del viaje en minutos

## Objetivo del modelo
Predecir el tiempo_min (regresion) a partir de las variables explicativas.
