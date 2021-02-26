REGISTRY=docker.belco.tech
TAG=$(shell git rev-parse --abbrev-ref HEAD)

docker-build:
	docker build -f docker/Dockerfile -t ${REGISTRY}/eegworkflow:${TAG} .

docker-run:
	docker run -it \
		   -v `pwd`/volume:/app/volume \
		   ${REGISTRY}/eegworkflow:${TAG} \
		   sws-pilot-workflow.py -h

test: test-types test-linting test-units

test-types:
	mypy --ignore-missing-imports --strict eegwlib scripts

test-linting:
	flake8 eegwlib scripts

test-units:
	pytest --cov
