FROM python:3.11

USER 0

WORKDIR /data
#ENV HOME /data
COPY . /data

RUN pip install --upgrade pip && \
    mkdir /docs && \
    chmod -R 775 /data && \
    chmod -R 775 /docs && \
    chmod -R 775 /usr && \
    pip --no-cache-dir install -r requirements.txt && \
    set -ex 

CMD ["streamlit", "run", "bot2.py", "--server.port=8080"]
