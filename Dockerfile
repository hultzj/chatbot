FROM python:3.11

ARG KUBE_LIST="https://kodekloud.com/wp-content/uploads/2020/11/Kubernetes-for-Beginners.pdf -O index.pdf"

WORKDIR /data

COPY . /data

RUN pip install --upgrade pip && \
    chmod -R 775 /data && \
    pip --no-cache-dir install -r requirements.txt && \
    set -ex && \
    wget ${KUBE_LIST}

CMD ["streamlit", "run", "bot.py", "--server.port=8080"]
