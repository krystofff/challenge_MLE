PY ?= python
DATA ?= data/data.csv
MODEL_OUT ?= models/delay_model.pkl

.ONESHELL:
ENV_PREFIX=$(shell python -c "if __import__('pathlib').Path('.venv/bin/pip').exists(): print('.venv/bin/')")

.PHONY: help
help:             	## Show the help.
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@fgrep "##" Makefile | fgrep -v fgrep

.PHONY: venv
venv:			## Create a virtual environment
	@echo "Creating virtualenv ..."
	@rm -rf .venv
	@python3 -m venv .venv
	@./.venv/bin/pip install -U pip
	@echo
	@echo "Run 'source .venv/bin/activate' to enable the environment"

.PHONY: install
install:		## Install dependencies
	pip install -r requirements-dev.txt
	pip install -r requirements-test.txt
	pip install -r requirements.txt

STRESS_URL = http://127.0.0.1:8000 
.PHONY: stress-test
stress-test:
	# change stress url to your deployed app 
	mkdir reports || true
	locust -f tests/stress/api_stress.py --print-stats --html reports/stress-test.html --run-time 60s --headless --users 100 --spawn-rate 1 -H $(STRESS_URL)

.PHONY: model-test
model-test:			## Run tests and coverage
	mkdir reports || true
	pytest --cov-config=.coveragerc --cov-report term --cov-report html:reports/html --cov-report xml:reports/coverage.xml --junitxml=reports/junit.xml --cov=challenge tests/model

.PHONY: api-test
api-test:			## Run tests and coverage
	mkdir reports || true
	pytest --cov-config=.coveragerc --cov-report term --cov-report html:reports/html --cov-report xml:reports/coverage.xml --junitxml=reports/junit.xml --cov=challenge tests/api

.PHONY: build
build:			## Build locally the python artifact
	python setup.py bdist_wheel

.PHONY: lint
lint: ## Ruff lint
	ruff check .

.PHONY: format
format: ## Ruff format (auto-fix)
	ruff format .

# -------- Training --------
.PHONY: train
train: 
	$(PY) challenge/train.py --data $(DATA) --output $(MODEL_OUT)

.PHONY: clean-model
clean-model:
	rm -f $(MODEL_OUT)

.PHONY: docker-build
docker-build:
	test -f $(MODEL_OUT) || (echo ">> Missing $(MODEL_OUT). Run 'make train' first." && false)
	docker build -t latam-delay-api:local .

.PHONY: docker-run
docker-run:
	docker run --rm -p 8000:8000 -e MODEL_VERSION=local latam-delay-api:local
