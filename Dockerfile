# syntax=docker/dockerfile:1

FROM ubuntu:16.04

MAINTAINER amal "amal.chandrasekhar@gmail.com"

RUN apt-get update -y && \
apt-get install -y python3-pip python-dev libsndfile1

EXPOSE 80
EXPOSE 5000

COPY ./requirements.txt /audio_analyzer/requirements.txt

WORKDIR /audio_analyzer

RUN pip3 install -r requirements.txt

COPY . /audio_analyzer

ENTRYPOINT [ "python3" ]

CMD [ "app.py" ]