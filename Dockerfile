FROM python:3.11-slim

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data/ data/
COPY src/ src/
COPY app/ app/

# Train at build time so the image is self-contained and reproducible --
# no model binary is committed to the repo, the Dockerfile is the source of truth.
RUN python src/train.py --data data/sms.tsv --out model

# Hugging Face Spaces (Docker SDK) expects the app to listen on port 7860.
EXPOSE 7860
ENV MODEL_PATH=model/model.joblib

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
