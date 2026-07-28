.PHONY: install test demo demo-neural clean

install:
	python -m pip install -e '.[dev,analysis]'

test:
	pytest -q

demo:
	genome demo --output artifacts/demo --force

demo-neural:
	genome demo --output artifacts/demo_neural --neural --force

clean:
	rm -rf .pytest_cache build dist *.egg-info
