REGISTRY=docker.belco.tech
TAG=$(shell git rev-parse --abbrev-ref HEAD)

build:
	docker build -f docker/Dockerfile -t ${REGISTRY}/eegworkflow:${TAG} .

run:
	docker run -it \
		   -v `pwd`/volume:/app/volume \
		   ${REGISTRY}/eegworkflow:${TAG} \
		   sws-pilot-workflow.py
