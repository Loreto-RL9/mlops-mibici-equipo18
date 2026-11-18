# Portabilidad del servicio (Entrega Final)

**Pregunta:** ¿el servicio da los mismos pronósticos en cualquier máquina y sistema operativo?

**Cómo se prueba:** el *golden set* (`reports/golden_set.json`) es un lote fijo de 122 filas estación-hora del
martes 15 de septiembre de 2026:
- 6 estaciones, de la de mayor a la de menor demanda, en sus 20 horas de servicio;
- 2 filas que deben caer en *fallback*: una estación desconocida y una fila sin rezago semanal.

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
| 18-nov-2026 | Alonso (Data Engineer) | Linux (Ubuntu, Python 3.12) | entorno de la imagen sin dependencias de desarrollo, usuario sin root, 2 workers | — | ✅ 122 filas, diferencia máx. 5e-11 |
| — | CI de GitHub (`contenedor.yml`) | ubuntu-latest | Podman | `mibici-api` con modelo sintético | ✅ `/health` y `/predict`, usuario 10001 |
| *pendiente* | César (DevOps / SRE) | Windows | Podman Desktop | `mibici-api` | |
| *pendiente* | José Luis (ML Engineer) | Windows o macOS | Podman | `mibici-api` | |

> Si el `make pipeline` de otra máquina registra un campeón con números distintos, `make smoke` lo detecta.
> Eso ya sería un hallazgo de reproducibilidad del entrenamiento, no del servicio: anotarlo aquí.
