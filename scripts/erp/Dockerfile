FROM python:3.8.8

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app

COPY erp.py .
COPY setup.py .
RUN pip install .

WORKDIR /volume

ENTRYPOINT ["erp.py"]
