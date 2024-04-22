FROM python:3.8-slim
WORKDIR /app
COPY . /app
# Install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg
RUN pip install -r requirements-prod.txt
EXPOSE 5000
ENV FLASK_APP=src/app/app.py
ENV FLASK_RUN_HOST=0.0.0.0
CMD ["python", "src/app/app.py"]
