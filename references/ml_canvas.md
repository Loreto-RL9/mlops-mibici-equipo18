# ML Canvas: demanda horaria por estación de MiBici

**Propuesta de valor.** Anticipar cuántas bicicletas saldrán de cada estación en las próximas horas, para que el
equipo de operación de MiBici rebalancee **antes** de que una estación se quede vacía. El usuario encuentra bici
cuando la necesita y el operador usa sus camiones donde más se requieren.

| Bloque | Contenido |
|---|---|
| **Tarea de predicción** | Regresión de conteos sobre series de tiempo de muchas estaciones: número de salidas de la estación *e* en la hora *h* (horizonte de 1 a 24 h). |
| **Decisiones** | Plan horario de rebalanceo: priorizar las estaciones con demanda prevista mayor que las bicis disponibles, asignar rutas a los camiones y programar el mantenimiento en horas valle. |
| **Simulación de impacto** | Antes de desplegar, comparar contra una línea base de persistencia semanal (misma hora de la semana anterior). Subestimar deja estaciones vacías (viajes perdidos); sobreestimar genera viajes de camión innecesarios. Meta de la Fase 2: bajar el MAE al menos 10 % frente a esa línea base. |
| **Hacer predicciones** | Por lotes, cada hora (o cada noche para las 24 h siguientes). Una latencia de minutos es suficiente. |
| **Construir modelos** | Reentrenar cada mes, cuando se publican nuevos datos abiertos, con partición temporal (pasado para entrenar, futuro para evaluar). Cada dataset, parámetro y métrica queda registrado en MLflow. |
| **Recolección de datos** | Publicación mensual de datos abiertos (con alrededor de un mes de retraso). Para operar en tiempo real haría falta un feed de disponibilidad de estaciones, por ejemplo con el estándar GBFS. |
| **Fuentes de datos** | 12 CSV mensuales de viajes (oct 2025 a sep 2026, 4.5 millones de viajes), catálogo de 484 estaciones y calendario oficial de días festivos de México. A futuro: clima y eventos. |
| **Variables** | Hora, día de la semana, fin de semana, festivo, codificación cíclica de hora y día, capacidad, ubicación, municipio, rezagos de 1, 24 y 168 h, media de las 24 h previas y componentes PCA del perfil horario de la estación. |
| **Monitoreo** | MAE por estación y por hora frente a la línea base, deriva de la demanda (estaciones nuevas o retiradas, eventos) y alertas cuando el error del modelo supere al de la línea base. |
