FROM python:3.10-slim

# تثبيت مكتبات النظام المطلوبة لـ OpenCV (التوافق مع trixie)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements_of_api.txt .
RUN pip install --no-cache-dir -r requirements_of_api.txt
COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]