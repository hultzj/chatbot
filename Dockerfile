from python:3.11
USER 0
RUN pip install --upgrade pip
WORKDIR /tmp
COPY . /tmp

RUN pip --no-cache-dir install -r requirements.txt

ENTRYPOINT ["python3"]
CMD ["/tmp/bot.py"]