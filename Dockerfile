FROM python:3.11

USER 0

WORKDIR /data

COPY . /data

RUN pip install --upgrade pip && \
    mkdir /docs && \
    chmod -R 775 /data && \
    chmod -R 775 /docs && \
    pip --no-cache-dir install --user -r requirements.txt && \
    set -ex 
  #  wget https://kodekloud.com/wp-content/uploads/2020/11/Kubernetes-for-Beginners.pdf --output-document=/docs/index.pdf

CMD ["streamlit", "run", "bot2.py", "--server.port=8080"]
