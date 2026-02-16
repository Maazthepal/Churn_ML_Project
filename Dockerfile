# Use a slim Python image (smaller, faster)
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose the port (Koyeb will use $PORT, but this documents it)
EXPOSE 8000

# Run Gunicorn with dynamic $PORT binding
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "app:app"]