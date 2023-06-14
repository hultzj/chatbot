from registry.access.redhat.com/ubi8/python-36

WORKDIR /app
COPY . /app

RUN pip --no-cache-dir install -r requirements.txt

ENTRYPOINT ["python3"]
CMD ["bot.py"]