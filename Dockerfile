# Use a base image from AzureML
FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3-pip
RUN pip install --upgrade pip

# Copy the necessary files into the container
COPY ./mlflow_model /mlflow_model

# Install the model dependencies
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Configure the MLflow model (if needed)
RUN pip install mlflow==2.19.0
RUN pip install cloudpickle==3.1.1

# Expose the port for the API (if it's a web model, otherwise this step can be omitted)
EXPOSE 5001

# Start the MLflow server (or another service if necessary)
CMD ["mlflow", "models", "serve", "--model-uri", "/mlflow_model", "--host", "0.0.0.0", "--port", "5001"]