.PHONY: all
UV ?= uv

all: pre-commit test
.PHONY: pre-commit
pre-commit:
	pre-commit run --all-files
.PHONY: test
test:
	coverage run -m unittest discover -s tests
	coverage report
	coverage html

.PHONY: clean
clean:
	coverage erase
	rm -rf htmlcov
	rm -rf build dist *.egg-info

.PHONY: build
build: clean
	$(UV) run --no-project --with build python -m build

.PHONY: push
push:
	$(UV) run --no-project --with twine python -m twine upload dist/*

.PHONY: open-html
open-html:
	open htmlcov/index.html

.PHONY: format
format:
	python3 -m ruff format .

.PHONY: mypy
mypy:
	python3 -m mypy optimed tests

.PHONY: check
check:
	python3 -m ruff check .
	$(MAKE) mypy
