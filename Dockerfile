# Use the official Python image from the Docker Hub
FROM python:3.9

# Set the working directory inside the container
WORKDIR /code

# Copy the requirements file into the container
COPY ./requirements.txt /code/requirements.txt

RUN apt-get update && apt-get install -y \
    libhdf5-dev
# RUN apt-get install -y libgl1-mesa-glx --fix-missing

# Install the dependencies
RUN pip install --no-cache-dir -r /code/requirements.txt

# Copy the application code into the container
COPY ./app /code/app

# Expose the port FastAPI will run on
EXPOSE 8080

# Set the entry point command to run FastAPI using uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
