FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin appuser

COPY --chown=appuser:appuser . .
ENV HF_HOME=/home/appuser/.cache/huggingface
USER appuser

# Exactly one model process; two HTTP threads allow fast 429 responses while
# the single bounded generation slot is occupied.
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 1 --threads 2 --timeout 120 'app:create_app()'"]
