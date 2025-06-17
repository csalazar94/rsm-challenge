FROM python:3.12-slim AS build

RUN apt-get update
RUN apt-get install -y libpq-dev gcc libgmp-dev libmpfr-dev libmpc-dev

RUN python3 -m venv /venv
ENV PATH=/venv/bin:$PATH

WORKDIR /code
COPY requirements.txt .
RUN pip install --no-cache-dir -r /code/requirements.txt

FROM python:3.12-slim

RUN apt-get update
RUN apt-get install -y libmagic-dev poppler-utils tesseract-ocr libpq-dev libgl1
RUN apt-get autoremove -y
RUN apt-get clean -y
RUN rm -rf /var/lib/apt/lists/*

COPY --from=build /venv /venv
ENV PATH=/venv/bin:$PATH

RUN python -m nltk.downloader punkt

WORKDIR /code
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3000"]
