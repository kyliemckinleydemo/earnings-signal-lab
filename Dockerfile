FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory (Railway volume will mount here)
RUN mkdir -p earnings_signal_data/transcripts earnings_signal_data/analysis

# Expose port
EXPOSE 8000

# Run the web server
CMD ["python", "server/app.py"]
