FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

WORKDIR /llm1

# 빌드 필수 도구
#RUN apt-get update && apt-get install -y build-essential git cmake && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git cmake \
    libxml2-dev libxslt1-dev libjpeg-dev zlib1g-dev \
 && rm -rf /var/lib/apt/lists/*

# # 라이브러리 설치
RUN pip install --upgrade pip setuptools wheel

COPY requirements.txt /llm1/requirements.txt
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

