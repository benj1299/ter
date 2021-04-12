FROM python:3

RUN apt-get update -y && \
    apt-get install -y locales && \
    apt-get install -y gcc && \
    apt-get install -y python3-sklearn python3-sklearn-lib ffmpeg libsm6 libxext6

RUN sed -i -e 's/# fr_FR.UTF-8 UTF-8/fr_FR.UTF-8 UTF-8/' /etc/locale.gen && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=fr_FR.UTF-8

ENV LANG fr_FR.UTF-8

WORKDIR /usr/src/app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt
