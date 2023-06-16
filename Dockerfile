from python:3.11
RUN pip install --upgrade pip
WORKDIR /data
COPY . /data
RUN chmod -R 775 /data

RUN pip --no-cache-dir install -r requirements.txt

ARG KUBE_LIST="\
  https://kodekloud.com/wp-content/uploads/2020/11/Kubernetes-for-Beginners.pdf -O index.pdf
"

RUN set -ex \
  && wget ${KUBE_LIST}

ENTRYPOINT ["python3"]
CMD ["bot.py"]