from python:3.11
RUN pip install --upgrade pip
WORKDIR /data
COPY . /data
RUN chmod -R 775 /data

RUN pip --no-cache-dir install -r requirements.txt

ENTRYPOINT ["python3"]
CMD ["bot.py"]