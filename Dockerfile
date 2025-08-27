FROM python:3.9-slim

WORKDIR /app

# Install only necessary system dependencies
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
RUN python -c "import moviepy.editor; print('MoviePy import OK')"

# Copy the rest of the app
COPY . .

# Create streamlit directory (but don't copy secrets.toml - Render handles this)
RUN mkdir -p /app/.streamlit

# NOTE: secrets.toml will be provided by Render's environment variables or secret files
# No need to COPY it here - it causes build failures since it's not in the repo

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "DIGITAL_COMPANION_APP.py", "--server.port=8501", "--server.address=0.0.0.0"]