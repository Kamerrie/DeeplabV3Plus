# Use the official TensorFlow GPU image
FROM tensorflow/tensorflow:2.16.1-gpu

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt /app/

RUN apt-get update && apt-get install -y \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the script and any other necessary files to the container
COPY evaluate-disease-dlv3plus.py /app/

# Settings some env vars for tensorflow, logs, and memory management
ENV TF_GPU_ALLOCATOR=cuda_malloc_async
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

# Command to run your Python script
CMD ["python", "evaluate-disease-dlv3plus.py"]
