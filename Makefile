#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = mlops-mibici-equipo18
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run tests
.PHONY: test
test:
	python -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	



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

## Panel de demanda estación x hora (v2) y dataset listo para modelar (v3)
.PHONY: features
features:
	$(PYTHON_INTERPRETER) -m mibici.features

## Generar las figuras del EDA en reports/figures
.PHONY: plots
plots:
	$(PYTHON_INTERPRETER) -m mibici.plots

## Líneas base sobre v3 (experimento mibici-fase1-baseline)
.PHONY: baseline
baseline:
	$(PYTHON_INTERPRETER) -m mibici.modeling.train

## Pipeline completo de la Fase 1
.PHONY: pipeline
pipeline: download data features plots baseline

## Abrir la interfaz de MLflow en http://127.0.0.1:8080
.PHONY: mlflow-ui
mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 8080


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
