FROM python:3.8.8

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app

COPY eegwlib eegwlib
COPY scripts scripts
COPY setup.py .
RUN pip install .

COPY docker/entrypoint.sh .

ENTRYPOINT ["/bin/bash", "entrypoint.sh"]
