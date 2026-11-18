#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = mlops-mibici-equipo18
# Con uv, cada comando corre en el entorno de uv.lock (sin activar .venv); sin uv, con el python
# del entorno activo
ifneq ($(shell command -v uv 2>/dev/null),)
RUN = uv run --frozen
PYTHON_INTERPRETER = $(RUN) python
else
RUN =
PYTHON_INTERPRETER = python
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies (versiones fijadas en uv.lock; sin uv, con pip y requirements.txt)
.PHONY: requirements
requirements:
ifneq ($(RUN),)
	uv sync --frozen
else
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
endif


## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	$(RUN) ruff format --check
	$(RUN) ruff check

## Format source code with ruff
.PHONY: format
format:
	$(RUN) ruff check --fix
	$(RUN) ruff format



## Run tests
.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m pytest tests

## Recuperar la versión registrada desde el remoto DVC local C:\mlops\files
.PHONY: dvc-pull
dvc-pull:
	$(RUN) dvc pull

## Subir una nueva versión al remoto DVC configurado
.PHONY: dvc-push
dvc-push:
	$(RUN) dvc push


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Descargar los datos crudos de MiBici (12 meses + catálogo de estaciones)
.PHONY: download
download:
	$(PYTHON_INTERPRETER) -m mibici.dataset download

## Limpiar los viajes y versionar v0 (crudo) y v1 (limpio) en MLflow
.PHONY: data
data:
	$(PYTHON_INTERPRETER) -m mibici.dataset clean

## Panel de demanda estación-hora (v2) y entrada del modelo con partición temporal (v4)
.PHONY: features
features:
	$(PYTHON_INTERPRETER) -m mibici.features

## Generar las figuras del EDA en reports/figures
.PHONY: plots
plots:
	$(PYTHON_INTERPRETER) -m mibici.plots

## Comparar algoritmos, ajustar con Optuna y registrar los Pipelines en el Model Registry
.PHONY: tune
tune:
	$(PYTHON_INTERPRETER) -m mibici.modeling.tune

## Pronosticar septiembre 2026 con el modelo "champion" del Model Registry
.PHONY: predict
predict:
	$(PYTHON_INTERPRETER) -m mibici.modeling.predict

## Exportar el modelo @champion del Model Registry a models/champion (lo que sirve la API)
.PHONY: export-model
export-model:
	$(PYTHON_INTERPRETER) -m mibici.modeling.export

## Levantar la API en local: http://127.0.0.1:8000/docs (requiere make export-model)
.PHONY: serve
serve:
	$(RUN) uvicorn mibici.api:app --host 127.0.0.1 --port 8000

# Motor de contenedores: Podman si está instalado; si no, Docker
ENGINE ?= $(shell command -v podman 2>/dev/null || command -v docker 2>/dev/null)
IMAGE ?= mibici-api
# Podman guarda por omisión en formato OCI, que ignora el HEALTHCHECK; con --format docker se conserva
BUILD_FLAGS = $(if $(findstring podman,$(ENGINE)),--format docker,)

## Generar el golden set (lote fijo de septiembre con los pronósticos del campeón exportado)
.PHONY: golden
golden:
	$(PYTHON_INTERPRETER) -m mibici.smoke golden

## Construir la imagen del servicio (requiere make export-model)
.PHONY: image
image:
	@test -f models/champion/MLmodel || (echo "Falta models/champion: corre make export-model" && exit 1)
	$(ENGINE) build $(BUILD_FLAGS) -t $(IMAGE) -f Dockerfile \
		--label mibici.modelo.version=$(MODEL_VERSION) --label mibici.api.version=$(API_VERSION) .

## Correr el contenedor en http://127.0.0.1:8000 (Ctrl+C para detenerlo)
.PHONY: run-image
run-image:
	$(ENGINE) run --rm --name $(IMAGE) -p 8000:8000 $(IMAGE)

API_VERSION = $(shell $(PYTHON_INTERPRETER) -c "from mibici import config; print(config.API_VERSION)" 2>/dev/null)
MODEL_VERSION = $(shell python3 -c "import json; print(json.load(open('models/champion/model_meta.json'))['version'])" 2>/dev/null)
DOCKERHUB_USER ?=
REPO = docker.io/$(DOCKERHUB_USER)/mibici-api

## Publicar la imagen en Docker Hub: tags API-modelo, API y latest (requiere docker login)
.PHONY: publish
publish:
	@test -n "$(DOCKERHUB_USER)" || (echo "Uso: make publish DOCKERHUB_USER=<usuario>" && exit 1)
	$(ENGINE) tag $(IMAGE) $(REPO):$(API_VERSION)-modelo$(MODEL_VERSION)
	$(ENGINE) tag $(IMAGE) $(REPO):$(API_VERSION)
	$(ENGINE) tag $(IMAGE) $(REPO):latest
	$(ENGINE) push $(REPO):$(API_VERSION)-modelo$(MODEL_VERSION)
	$(ENGINE) push $(REPO):$(API_VERSION)
	$(ENGINE) push $(REPO):latest

## Exportar el esquema OpenAPI de la API a docs/openapi.json
.PHONY: openapi
openapi:
	$(PYTHON_INTERPRETER) -c "import json; from mibici.api import app; open('docs/openapi.json', 'w').write(json.dumps(app.openapi(), indent=2, ensure_ascii=False))"

## Comparar la API en marcha (make serve o make run-image) contra el golden set
.PHONY: smoke
smoke:
	$(PYTHON_INTERPRETER) -m mibici.smoke check --url http://127.0.0.1:8000

## Monitoreo: drift de entradas, predicciones y demanda real contra validación, escenarios
## sintéticos, error mensual y alertas (resultados en reports/monitoreo y en MLflow)
.PHONY: drift
drift:
	$(PYTHON_INTERPRETER) -m mibici.monitoring drift

## Reportes HTML de Evidently sobre las mismas muestras (entorno aislado: Evidently no soporta pandas 3)
.PHONY: drift-evidently
drift-evidently:
	uv run --no-project --python 3.12 --with "evidently==0.7.23" --with "pandas<3" --with pyarrow \
		python mibici/monitoring_evidently.py reports/monitoreo

## Drift de las peticiones reales que guardó la API en logs/predicciones.jsonl
.PHONY: drift-log
drift-log:
	$(PYTHON_INTERPRETER) -m mibici.monitoring log

## Pipeline completo de punta a punta (primero corre las pruebas)
.PHONY: pipeline
pipeline: test download data features plots tune predict

## Abrir la interfaz de MLflow en http://127.0.0.1:8080
.PHONY: mlflow-ui
mlflow-ui:
	$(RUN) mlflow ui --backend-store-uri sqlite:///mlflow.db --port 8080


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
