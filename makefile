test: test-types test-linting test-units

test-types:
	mypy --ignore-missing-imports --allow-untyped-decorators \
	--strict eegwlib

test-linting:
	flake8 eegwlib

test-units:
	pytest --cov
