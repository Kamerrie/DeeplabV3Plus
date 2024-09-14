FROM tensorflow/tensorflow:2.16.1-gpu

WORKDIR /app

COPY requirements.txt /app/

RUN apt-get update && apt-get install -y \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install seaborn
COPY evaluate-disease-dlv3plus.py /app/

ENV TF_GPU_ALLOCATOR=cuda_malloc_async
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

CMD ["python", "evaluate-disease-dlv3plus.py"]
