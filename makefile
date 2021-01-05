REGISTRY=docker.belco.tech
TAG=$(shell git rev-parse --abbrev-ref HEAD)

docker-build:
	docker build -f docker/Dockerfile -t ${REGISTRY}/eegworkflow:${TAG} .

docker-run:
	docker run -it \
		   -v `pwd`/volume:/app/volume \
		   ${REGISTRY}/eegworkflow:${TAG} \
		   sws-pilot-workflow.py -h

test:
	flake8 eegwlib
	mypy --ignore-missing-imports --strict eegwlib
	pytest --cov
