# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install only the necessary system dependencies
RUN apt-get update \
 && apt-get install -y \
    gcc \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
 && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
 && python -m pip install --no-cache-dir -r requirements.txt \
 && python -m pip install --no-cache-dir moviepy imageio-ffmpeg pillow

# Sanity check for MoviePy import
RUN python - <<EOF
import moviepy.editor
print("MoviePy import OK")
EOF

# Copy the rest of the app
COPY . .

# Streamlit config
RUN mkdir -p /app/.streamlit
COPY .streamlit/secrets.toml /app/.streamlit/secrets.toml

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "DIGITAL_COMPANION_APP.py", "--server.port=8501", "--server.address=0.0.0.0"]
