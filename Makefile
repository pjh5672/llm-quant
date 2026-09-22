.PHONY: test quality style check

CHECK_DIRS := llmquant examples tests

test:
	python -m pytest -q

quality:
	python -m ruff check $(CHECK_DIRS)

style:
	python -m ruff check $(CHECK_DIRS) --fix

check: quality test
