FROM python:3.11

USER 0

WORKDIR /data

COPY . /data

RUN pip install --upgrade pip && \
    chmod -R 775 /data && \
    pip --no-cache-dir install -r requirements.txt && \
    set -ex && \
    wget https://kodekloud.com/wp-content/uploads/2020/11/Kubernetes-for-Beginners.pdf --output-document=/data/index.pdf

CMD ["streamlit", "run", "bot.py", "--server.port=8080"]
