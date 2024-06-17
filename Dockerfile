FROM python:3.10-buster

ADD . /opt/ml_in_app
WORKDIR /opt/ml_in_app

# install packages by conda
RUN pip install numpy==1.26.3
RUN pip install -r requirements_prod.txt
CMD ["python", "app.py"]
