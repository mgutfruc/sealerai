#FROM python:3.8.12 
FROM python:3.10
#python version 3.10.4

WORKDIR /backend

COPY requirements.txt /backend
#COPY requirements.txt requirements.txt

#ENV http_proxy http://internet.ford.com:83
#ENV https_proxy http://internet.ford.com:83
#ENV no_proxy "localhost, 127.0.0.1, .ford.com"

RUN pip3 install -r requirements.txt

RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python
#absolute pass
COPY . /backend 

#WORKDIR backend

EXPOSE 5001

ENV FLASK_APP=app.py

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]

#CMD [ "python3", "-m" , "app.py", "run", "--host=0.0.0.0"]
