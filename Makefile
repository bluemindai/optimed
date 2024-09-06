.PHONY: all
all: test

.PHONY: test
test:
	coverage run -m unittest discover -s tests
	coverage report
	coverage html

.PHONY: clean
clean:
	coverage erase
	rm -rf htmlcov

.PHONY: open-html
open-html:
	open htmlcov/index.html