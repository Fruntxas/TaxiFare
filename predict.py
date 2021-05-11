import pandas as pd
import joblib
from google.cloud import storage

BUCKET_NAME = 'wagon-ml-pereira-566'
STORAGE_LOCATION = 'model.joblib'

df_test = pd.read_csv("raw_data/test.csv")
print("df_test loaded")

#model = joblib.load("Models/GradientBoostingRegressor().joblib")
#print("Model loaded")

client = storage.Client()
bucket = client.get_bucket(BUCKET_NAME)
blob = bucket.blob(STORAGE_LOCATION)

blob.download_to_filename('teste.joblib')

model = joblib.load('teste.joblib')

#model = joblib.load('model.joblib')

#print("Model loaded from Storage")

pred = model.predict(df_test)
print("Prediction created from GCP model")

df_test['fare_amount'] = pred

df_results = df_test[["key","fare_amount"]]

#df_results.to_csv("submission.csv",index=False)
print("Results saved to disk")