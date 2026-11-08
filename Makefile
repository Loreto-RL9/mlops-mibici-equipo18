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
