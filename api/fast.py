from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"Greetings": "Welcome to my Taxifare Model Predictor"}

@app.get("/predict_fare")
def predict_fare(key,pickup_datetime,pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,passenger_count):

    #url = 'http://localhost:8000/predict_fare'

    params = {
    "key" : key,
    "pickup_datetime": pickup_datetime,
    "pickup_longitude": pickup_longitude,
    "pickup_latitude": pickup_latitude,
    "dropoff_longitude": dropoff_longitude,
    "dropoff_latitude": dropoff_latitude,
    "passenger_count": passenger_count
    }

    X_pred = pd.DataFrame([params.values()],columns=params.keys())

    # Load Model Locally
    model = joblib.load('teste.joblib')

    # Load Model from GCP
    # from google.cloud import storage

    # BUCKET_NAME = 'wagon-ml-pereira-566'
    # STORAGE_LOCATION = 'model.joblib'

    # client = storage.Client()
    # bucket = client.get_bucket(BUCKET_NAME)
    # blob = bucket.blob(STORAGE_LOCATION)

    # blob.download_to_filename('teste.joblib')

    # model = joblib.load('teste.joblib')

    pred = int(model.predict(X_pred))

    return pred
    #=> {wait: 64}
