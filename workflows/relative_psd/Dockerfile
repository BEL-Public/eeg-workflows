FROM python:3.8.8

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app

COPY relative_psd.py .
COPY setup.py .
RUN pip install .

WORKDIR /volume

ENTRYPOINT ["relative_psd.py"]
