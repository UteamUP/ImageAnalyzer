.PHONY: install test lint run dry-run clean status

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

install:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt
	@echo ""
	@echo "Virtual environment created. Activate with:"
	@echo "  source $(VENV)/bin/activate"

test:
	$(PYTHON) -m pytest tests/ -v --cov=src/image_analyzer --cov-report=term-missing

lint:
	$(PYTHON) -m py_compile src/image_analyzer/*.py
	$(PYTHON) -m py_compile src/image_analyzer/utils/*.py

run:
	$(PYTHON) -m image_analyzer analyze

dry-run:
	$(PYTHON) -m image_analyzer analyze --dry-run

status:
	$(PYTHON) -m image_analyzer status

clean:
	rm -rf $(VENV) .checkpoint.json .checkpoint.json.tmp .checkpoint.lock
	rm -rf __pycache__ src/image_analyzer/__pycache__ src/image_analyzer/utils/__pycache__
	rm -rf .pytest_cache htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
