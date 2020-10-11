FROM python:3.6.7

ADD requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app
COPY scripts scripts
COPY docker/entrypoint.sh .

ENTRYPOINT ["/bin/bash", "entrypoint.sh"]
