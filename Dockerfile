FROM python:3.12-slim

# Build dependencies for pybullet and pycapnp
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    g++ \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Default: run training
CMD ["python", "tools/sim/train_policy.py", "--curriculum", "--timesteps", "20000000"]
