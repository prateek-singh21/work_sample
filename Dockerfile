# Use an official Python runtime as a parent image
FROM python:3.10.8

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Set environment variables
ENV AIRFLOW_HOME=/app/airflow
ENV PYTHONPATH=/app

# Expose the ports
EXPOSE 8080

# Run the command to start Airflow webserver and scheduler
CMD ["airflow", "webserver", "-p", "8080"]

