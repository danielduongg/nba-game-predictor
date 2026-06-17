FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# default: fast end-to-end run (skips hyperparameter search)
CMD ["python", "main.py", "--no-tune"]
