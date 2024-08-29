# Use the official TensorFlow GPU image
FROM tensorflow/tensorflow:2.16.1-gpu

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt /app/

RUN apt-get install -y git
# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the script and any other necessary files to the container
COPY disease-dlv3plus.py /app/

# Command to run your Python script
CMD ["python", "disease-dlv3plus.py"]
