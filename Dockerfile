FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# only the files the service needs (no tests, no data, no secrets)
COPY work_passport.py llm_tools.py generate_synthetic_data.py scoring_api.py ./

# most hosts inject PORT; default to 8000
ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT','8000'))"

CMD ["sh", "-c", "uvicorn scoring_api:app --host 0.0.0.0 --port ${PORT}"]
