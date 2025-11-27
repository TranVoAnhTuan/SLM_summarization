# ----------------------------
# 1. Chọn image Python chính thức
# ----------------------------
FROM python:3.10-slim

# ----------------------------
# 2. Đặt thư mục làm việc
# ----------------------------
WORKDIR /app

# ----------------------------
# 3. Copy toàn bộ project vào container
# ----------------------------
COPY . /app

# ----------------------------
# 4. Cài đặt các dependency
# ----------------------------
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# ----------------------------
# 5. Expose port Django server
# ----------------------------
EXPOSE 8000

# ----------------------------
# 6. Lệnh khởi động server
# ----------------------------
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
