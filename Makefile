# Convenience Makefile for ThoughtTracker ML.
# Uses public PyPI explicitly so a corporate pip.conf doesn't break installs.

PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

PIP_INDEX ?= https://pypi.org/simple/

.PHONY: help venv install train evaluate predict serve test clean

help:
	@echo "Targets:"
	@echo "  make install   - create venv and install requirements from public PyPI"
	@echo "  make train     - fine-tune DistilBERT on DATASET_PATH"
	@echo "  make evaluate  - re-evaluate the saved model on the test split"
	@echo "  make predict   - run a CLI prediction (set TOPIC= and TEXT=)"
	@echo "  make serve     - run the FastAPI server on port 8000"
	@echo "  make test      - run pytest"
	@echo "  make clean     - remove venv, model artifacts, and reports"

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)

venv: $(VENV)/bin/activate

install: venv
	$(PIP) install --index-url $(PIP_INDEX) --upgrade pip
	$(PIP) install --index-url $(PIP_INDEX) -r requirements.txt

train:
	$(PY) -m src.training.train

evaluate:
	$(PY) -m src.training.evaluate

predict:
	@if [ -z "$(TOPIC)" ] || [ -z "$(TEXT)" ]; then \
		echo "Usage: make predict TOPIC='ai' TEXT='I support this and I am in favor.'"; \
		exit 1; \
	fi
	$(PY) -m src.inference.predict --topic "$(TOPIC)" --text "$(TEXT)"

serve:
	$(PY) -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

test:
	$(PY) -m pytest -q

clean:
	rm -rf $(VENV) models/stance-classifier reports/figures/*.png reports/metrics/*.json
