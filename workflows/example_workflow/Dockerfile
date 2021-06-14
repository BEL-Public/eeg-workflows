FROM python:3.8.8

WORKDIR /app

COPY example.py .
COPY setup.py .
RUN pip install .

WORKDIR /volume

ENTRYPOINT ["example.py"]
