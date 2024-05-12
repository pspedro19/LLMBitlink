FROM python:3.8-slim

WORKDIR /app
COPY . .
COPY requirements.txt /requirements.txt
ENV OPENAI_API_KEY=c4129d4dad0247a691ad29bd38db1260
ENV OPENAI_API_BASE=https://openaiuniandesdevtest.openai.azure.com/
ENV OPENAI_API_VERSION=2023-05-15
ENV OPENAI_ENGINE=gpt-turbo-RF-NT

RUN apt-get update && apt-get install -y curl
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /requirements.txt

EXPOSE 8800
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
