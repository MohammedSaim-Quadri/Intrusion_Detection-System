# Use an official TensorFlow base image with Python 3 support
FROM tensorflow/tensorflow:2.16.1

# Set the working directory inside the container
WORKDIR /app

# Copy only the essential files to avoid unnecessary dependencies
COPY ./src /app/src
COPY ./artifacts /app/artifacts
COPY ./dataset /app/dataset
COPY requirements.txt /app

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose a port for monitoring (TensorBoard or other tools)
EXPOSE 6006

# Set the default command to run your Optuna tuning script
CMD ["python", "src/components/data_ingestion.py"]
