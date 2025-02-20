.PHONY: all
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
	python3 -m build

.PHONY: push
push:
	python3 -m twine upload dist/*

.PHONY: open-html
open-html:
	open htmlcov/index.html
