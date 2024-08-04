# syntax=docker/dockerfile:1

FROM python:3.8

ENV LANG C.UTF-8
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install dependencies and create a virtual environment
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        python3-dev \
        git \
        libfftw3-dev \
        libavcodec-dev \
        libavformat-dev \
        libavresample-dev \
        libsamplerate0-dev \
        libtag1-dev \
        libyaml-dev \
        ffmpeg \
        htop \
        vim \
    && python -m venv $VIRTUAL_ENV \
    && pip install --upgrade pip \
    && pip install virtualenv \
    && pip install librosa fire \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt before copying the entire project
COPY ./requirements.txt /audio_analyzer/requirements.txt

WORKDIR /audio_analyzer

RUN pip install -r requirements.txt

# Copy the rest of the application code
COPY . /audio_analyzer

# Set entrypoint and default command
ENTRYPOINT ["python3"]
CMD ["app.py"]
