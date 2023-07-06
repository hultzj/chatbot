FROM python:3.11

USER 0

WORKDIR /data

COPY . /data

RUN pip install --upgrade pip && \
    chmod -R 775 /data && \
    pip --no-cache-dir install -r requirements.txt && \
    set -ex 

CMD ["streamlit", "run", "main.py", "--server.port=8080"]
