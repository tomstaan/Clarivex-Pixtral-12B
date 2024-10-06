# Use the nvidia/cuda base image (devel version)
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# Set the working directory in the container
WORKDIR /app

# Install Git and necessary dependencies
RUN apt-get update && apt-get install -y git python3-pip

# Install 'packaging' and 'torch'
RUN pip3 install packaging
RUN pip3 install torch

# Copy the application code into the container
COPY . /app

# Install all required Python packages from the requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Expose port 5000
EXPOSE 5000

# Define the command to run the app
CMD ["python3", "app.py"]

