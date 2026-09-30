PYTHON ?= python

.PHONY: install install-dev test test-integration lint format benchmark-validate benchmark-old benchmark-new run-v1 run-v2 worker-test check

install:
	$(PYTHON) -m pip install -e ".[runtime]"

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests/unit

test-integration:
	NCAIR_RUN_INTEGRATION=1 $(PYTHON) -m pytest -m integration tests/integration

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

benchmark-validate:
	$(PYTHON) -m eval.validate_benchmark

benchmark-old:
	$(PYTHON) -m eval.run_benchmark --version v1 --mode routing

benchmark-new:
	$(PYTHON) -m eval.run_benchmark --version v2 --mode routing

run-v1:
	NCAIR_DEFAULT_VERSION=v1 uvicorn ncair_lms.api:app --reload

run-v2:
	NCAIR_DEFAULT_VERSION=v2 uvicorn ncair_lms.api:app --reload

worker-test:
	cd worker && npm test

check: lint benchmark-validate test worker-test
