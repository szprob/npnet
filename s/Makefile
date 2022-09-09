SHELL=/bin/bash
PROJECT_NAME=sprite

setup-dev:
	apt-get update && apt-get install clang-format -y
	pip3 install -r requirements-compile.txt
	pip3 install -r requirements-lint.txt
	pip3 install -r requirements.txt
	python setup.py install

check:
	find . -name *.cpp|xargs clang-format -i
	black --check .
	isort --check .
	flake8 .

test:
	pytest --cov=${PROJECT_NAME} tests/*
	coverage report


build:
	pip3 install cibuildwheel>=2.4.0 --user
	cibuildwheel --plat linux
	docker run --rm -v `pwd`/wheelhouse:/wheelhouse -v `pwd`/tests:/tests -e PYTHONDONTWRITEBYTECODE=1 python:3.7 sh -c "pip install `find . -name '*37*many*'` && pip3 install pytest==7.1.1 && pip3 install pytest-mock==1.10.4 && pytest tests -p no:cacheprovider"

upload:
	pip3 install twine
	twine upload  wheelhouse/*
