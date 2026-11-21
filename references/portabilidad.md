# Portabilidad del servicio (Entrega Final)

**Pregunta:** ¿el servicio da los mismos pronósticos en cualquier máquina y sistema operativo?

**Cómo se prueba:** el *golden set* (`reports/golden_set.json`) es un lote fijo de 123 filas estación-hora del
martes 15 de septiembre de 2026:
- 6 estaciones, de la de mayor a la de menor demanda, en sus 20 horas de servicio;
- 3 filas que deben caer en *fallback*: una estación desconocida, una fila sin rezago semanal y una estación nueva que
  el modelo no vio al entrenar.

Guarda los pronósticos que dio el campeón al exportarse. `make smoke` manda ese lote a la API en marcha y falla si algún
número cambia más de 1e-6 o si cambia la fuente (`modelo` o `fallback`).

## Pasos (cualquier sistema operativo)

```bash
git clone https://github.com/Loreto-RL9/mlops-mibici-equipo18 && cd mlops-mibici-equipo18
make requirements
make pipeline          # o, si ya tienes mlflow.db con el campeón, sólo: make export-model
make export-model
make image             # Podman (o Docker si no hay Podman)
make run-image         # en otra terminal:
make smoke
```

- **macOS:** Podman Desktop (o `brew install podman` y luego `podman machine init && podman machine start`).
- **Windows:** Podman Desktop con WSL2. Sin `make`, se corren los comandos que aparecen en el `Makefile`.

## Resultados

| Fecha | Quién (rol) | Sistema operativo | Motor | Imagen | `make smoke` |
|---|---|---|---|---|---|
| 18-nov-2026 | Alonso (Data Engineer) | Linux (Ubuntu, Python 3.12) | entorno de la imagen sin dependencias de desarrollo, usuario sin root, 2 workers | — | ✅ 123 filas, diferencia máx. 5e-11 |
| — | CI de GitHub (`contenedor.yml`) | ubuntu-latest | Podman | `mibici-api` con modelo sintético | ✅ `/health` y `/predict`, usuario 10001 |
| 21-nov-2026 | César (DevOps / SRE) | Linux (Ubuntu 26.04, x86_64, Intel i7-2620M) | Docker 29.1.3 | `cesarxyz/mibici-api:1.0.0-331c828` con el campeón reentrenado en esa máquina | ✅ 123 filas, diferencia máx. 4.9e-11 |
| *pendiente* | César (DevOps / SRE) | Windows | Podman Desktop | `mibici-api` con el campeón del repo | |
| *pendiente* | José Luis (ML Engineer) | Windows o macOS | Podman | `mibici-api` | |

> Si el `make pipeline` de otra máquina registra un campeón con números distintos, `make smoke` lo detecta.
> Eso ya sería un hallazgo de reproducibilidad del entrenamiento, no del servicio: anotarlo aquí.

## Reentrenamiento en otra máquina (César, 21-nov-2026)

Se corrió el pipeline completo desde cero en un servidor Linux sin `make` (pruebas, descarga, limpieza, features,
`tune`, `predict`, `export` y `golden`), con `uv sync --frozen` y las mismas versiones del `uv.lock`
(scikit-learn 1.9.1, pandas 3.1.1, numpy 2.5.4, Optuna 5.0.0, Python 3.12.16). Tardó 4 h 10 min en 4 núcleos.

**Datos: idénticos.** Las huellas sha256 de v1 (`806a17f8…`), v2 (`2a9817a1…`) y v4 (`c4247f81…`) coinciden con
[`reproducibilidad.md`](reproducibilidad.md), igual que los conteos. Sólo cambia la huella de v0, porque los CSV
crudos se volvieron a bajar del portal y sus bytes no son los mismos; después de la limpieza el resultado es igual.

**Modelo: distinto en los árboles.** La regresión de Poisson dio las mismas métricas a cuatro decimales; el bosque y
el HGB no:

| Modelo | MAE validación (repo → servidor) | MAE prueba (repo → servidor) |
|---|---:|---:|
| Regresión de Poisson | 1.2073 → 1.2073 | 1.3418 → 1.3418 |
| Bosque aleatorio | 0.9024 → 0.9019 | 1.0416 → 1.0412 |
| HGB Poisson (campeón) | 0.8654 → 0.8702 | 1.0054 → 1.0067 |

**Dónde se separa:** en el HGB, las pruebas 0 a 14 de Optuna dan los mismos hiperparámetros y el mismo score que la
gráfica `reports/figures/fase2_optuna_historial_hgb_poisson.png`. En la prueba 15 el sampler TPE propone otra
combinación (1.1440 en el repo, 1.1413 en el servidor) y desde ahí cada búsqueda sigue su camino.

**Causa probable:** diferencias de redondeo en la última cifra según el procesador (el i7-2620M no tiene AVX2 ni
FMA, así que numpy y BLAS usan otros kernels). Se descartó el número de hilos: con 1 y con 4 hilos el HGB da
exactamente lo mismo y el bosque difiere en 7e-15. La regresión de Poisson tiene un único óptimo y esas diferencias
no la mueven; en los árboles cambian qué hiperparámetros elige TPE.

**Conclusión:** el servicio es portable (el mismo modelo da los mismos números en el host y en el contenedor), pero la
búsqueda de hiperparámetros sólo se reproduce exactamente en el mismo tipo de máquina. Por eso la imagen oficial
debe llevar el artefacto fijo `@champion` con su `run_id`, no un reentrenamiento. La imagen publicada el 21-nov en
`cesarxyz/mibici-api` es provisional: lleva el campeón del servidor (`run_id` `331c828a…`) y se reconstruirá con el
del repo (`0c5f641f…`) para que coincida con `reports/golden_set.json`.
