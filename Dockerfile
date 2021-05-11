# write some code to build your image

FROM python:3.8.6-buster

COPY api /api
COPY requirements.txt /requirements.txt
COPY teste.joblib /teste.joblib
COPY TaxiFareModel /TaxiFareModel

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn api.fast:app --host 0.0.0.0

