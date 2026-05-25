.PHONY: app install predict test train tune

PYTHON ?= python3
CONFIG ?= configs/training.yaml
MODEL ?= models/heart_disease_pipeline.joblib
INPUT ?= data/raw/heart_sample.csv

install:
	$(PYTHON) -m pip install -e ".[dev]"

train:
	$(PYTHON) scripts/train.py --config $(CONFIG)

tune:
	$(PYTHON) scripts/tune.py --config $(CONFIG)

predict:
	$(PYTHON) scripts/predict.py --model $(MODEL) --input $(INPUT)

app:
	$(PYTHON) -m streamlit run src/heart_disease/web_app.py

test:
	$(PYTHON) -m pytest
