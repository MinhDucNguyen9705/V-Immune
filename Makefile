PY ?= python
VIMMUNE ?= vimmune

.PHONY: install check lint type test demo demo-interactive ui demo-clean eval

install:
	pip install -e ".[dev]"

lint:
	ruff check src tests

type:
	mypy

test:
	pytest

check: lint type test

# `make demo` gives the human approval via the CLI flag (spec §6.6 gate item 4).
# Use `make demo-interactive` to stop at the gate and approve from the Streamlit UI instead.
demo:
	$(VIMMUNE) demo run --profile demo --approve

demo-interactive:
	$(VIMMUNE) demo run --profile demo

ui:
	streamlit run src/vimmune/demo/app.py

demo-clean:
	$(VIMMUNE) demo clean

eval:
	$(VIMMUNE) eval loco --profile demo --folds 2 --out runs/eval && $(VIMMUNE) eval report --exp runs/eval
