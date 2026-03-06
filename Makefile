.PHONY: lint test

test:
	pytest -q

lint:
	ruff format --check .
	ruff check .
