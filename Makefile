.PHONY: install install-dev test lint collect analyze report clean

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest -v --tb=short

lint:
	ruff check arkprobe/ tests/

collect:
	arkprobe collect --scenario all

analyze:
	arkprobe analyze --input ./data

report:
	arkprobe report --output ./report.html

full:
	arkprobe full-run --output ./report.html

clean:
	rm -rf data/*.json data/*.data report.html
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
