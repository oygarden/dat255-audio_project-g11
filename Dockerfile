FROM python:3.8-slim
WORKDIR /usr/src/app
COPY ./src/app /usr/src/app
RUN pip install -r requirements.txt
EXPOSE 5000
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
CMD ["flask", "run"]
