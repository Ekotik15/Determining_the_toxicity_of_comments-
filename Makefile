.PHONY: install install-transformer test lint smoke train transformer analysis analysis-transformer app report

install:
	python -m pip install -r requirements-dev.txt

install-transformer:
	python -m pip install -r requirements-transformer.txt

test:
	python -m pytest

lint:
	python -m ruff check .
	python -m ruff format --check .

smoke:
	python scripts/run_classical.py --sample-size 2000 --max-features 5000 --artifacts-dir artifacts-smoke

train:
	python scripts/run_classical.py

transformer:
	python scripts/run_transformer.py \
		--epochs 1 \
		--max-length 128 \
		--train-batch-size 16 \
		--eval-batch-size 64 \
		--learning-rate 2e-5 \
		--weight-decay 0.01 \
		--random-state 42

analysis:
	python scripts/analyze_predictions.py

analysis-transformer:
	python scripts/analyze_predictions.py \
		--predictions artifacts/transformer_test_predictions.parquet \
		--metadata artifacts/transformer_metadata.json

app:
	streamlit run streamlit_app.py

report:
	python scripts/build_report.py
