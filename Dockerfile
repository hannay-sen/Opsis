FROM node:20-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app

# CPU-only PyTorch is much smaller than the default GPU build.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY data/index/ ./data/index/
COPY data/images/ ./data/images/
COPY --from=frontend /frontend/dist ./frontend/dist

# Hugging Face runs the container as a non-root user, so model downloads go to /tmp.
ENV HF_HOME=/tmp/hf \
    INDEX_DIR=/app/data/index \
    IMAGE_ROOT=/app/data/images \
    FRONTEND_DIST=/app/frontend/dist \
    CORS_ORIGINS=*

EXPOSE 7860
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]