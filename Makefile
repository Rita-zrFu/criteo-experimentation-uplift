PYTHON := .venv/bin/python

.PHONY: setup download prepare experiment heterogeneity uplift dml analysis all test

setup:
	python3 -m venv .venv
	$(PYTHON) -m pip install -r requirements-lock.txt

download:
	$(PYTHON) src/data.py download

prepare:
	$(PYTHON) src/data.py prepare --force

experiment:
	$(PYTHON) src/experiment.py

heterogeneity:
	$(PYTHON) src/heterogeneity.py

uplift:
	$(PYTHON) src/uplift.py

dml:
	$(PYTHON) src/dml.py

analysis: experiment heterogeneity uplift dml

all: prepare analysis

test:
	PYTHONPATH=. $(PYTHON) -m unittest discover -s tests -v
