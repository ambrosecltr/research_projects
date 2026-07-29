.PHONY: install test check help

install:
	python -m pip install -e '.[dev,evaluation]'

test:
	python -m pytest -q

check:
	python -m compileall -q genome tests
	python -m pytest -q

help:
	python -m genome --help
