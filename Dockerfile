FROM python:3.11-bullseye
COPY . /code/
WORKDIR /code
RUN pip install -r requirements.pip
ENTRYPOINT ["python3", "/main.py"]