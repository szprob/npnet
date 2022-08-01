SHELL=/bin/bash
PROJECT_NAME=sprite

setup-dev:
	pip install -r requirements-lint.txt
	pip install -r requirements-extension.txt
	pip install -r requirements.txt
	python setup.py install

check:
	black --check .
	isort --check .
	flake8 .

test:
	pytest tests/*

upload:
	pip install twine==4.0.0
	cibuildwheel --plat linux
	twine upload  wheelhouse/*
