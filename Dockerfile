# syntax=docker/dockerfile:1

FROM python:3.8

ENV LANG C.UTF-8

# pip libraries
RUN pip install --upgrade pip\
    && pip install virtualenv librosa fire

RUN apt-get update \
    && apt-get install -y build-essential python3-dev git \
    libfftw3-dev libavcodec-dev libavformat-dev libavresample-dev \
    libsamplerate0-dev libtag1-dev libyaml-dev

RUN apt-get update \
    && apt-get install -y ffmpeg \
    && apt-get install -y htop \
    && apt-get install -y vim

COPY ./requirements.txt /audio_analyzer/requirements.txt

WORKDIR /audio_analyzer

RUN pip3 install -r requirements.txt

COPY . /audio_analyzer

ENTRYPOINT [ "python3" ]

CMD [ "app.py" ]