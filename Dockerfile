FROM nvcr.io/nvidia/tritonserver:23.06-py3

RUN apt update && apt -y install libssl-dev tesseract-ocr libtesseract-dev ffmpeg

RUN pip install --upgrade pip

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY . /app
WORKDIR /app

CMD ["python3", "-u", "app.py"]