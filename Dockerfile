FROM python:3.9-slim AS builder

COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install -r requirements.txt

COPY . /app
CMD jupyter notebook --ip 0.0.0.0 --allow-root --no-browser covid_cognition.ipynb 
