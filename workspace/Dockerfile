FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

WORKDIR /app

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install -y python3-pip && pip3 install --upgrade pip

COPY . .

RUN pip3 install -r requirements.txt

CMD uvicorn main:app --host 0.0.0.0 --port 8000
