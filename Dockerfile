FROM tensorflow/tensorflow:latest
RUN pip install pillow Flask gunicorn gevent -i https://mirrors.aliyun.com/pypi/simple
COPY ./app /app
WORKDIR /app
CMD ["gunicorn", "webapi:app", "-w", "5","-b"," 0.0.0.0:5000","-k","gevent"]