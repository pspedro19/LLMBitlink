FROM python:3.8-slim

WORKDIR /app
COPY . .
COPY requirements.txt /requirements.txt
ENV OPENAI_API_KEY=sk-proj-XVAE4QMHvn69ktcWGIFnT3BlbkFJ5Gmqq3cFWkgMF6gCW1f7

RUN apt-get update && apt-get install -y curl
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /requirements.txt

EXPOSE 8800
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
