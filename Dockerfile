# Use official Python image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy all files from rice_classifier into the container
COPY . .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install -r rice_classifier/requirements.txt

# Expose the port your app runs on (Render uses 10000+ range internally)
EXPOSE 10000

# Start the Flask app using Gunicorn
CMD ["gunicorn", "rice_classifier.app:app", "--bind", "0.0.0.0:10000"]
