# imports
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

from TaxiFareModel.encoders import DistanceTransformer,TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data,clean_data, get_local_data

import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

from memoized_property import memoized_property

import joblib

MLFLOW_URI = "https://mlflow.lewagon.co/"
myname = "TiagoPereira"
EXPERIMENT_NAME = f"TaxifareModel_{myname}"

filepath = "/home/fruntxas/code/Fruntxas/TaxiFareModel/Models/"

class Trainer():

    def __init__(self, X, y, estimator):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.estimator = estimator

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        
        # create distance and time pipes
        pipe_distance = make_pipeline(DistanceTransformer(), StandardScaler())
        pipe_time = make_pipeline(TimeFeaturesEncoder(time_column='pickup_datetime'), OneHotEncoder())
        
        # set columns per pipe
        dist_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
        time_cols = ['pickup_datetime']

        # Column Transformer
        feat_eng_bloc = ColumnTransformer([('distance', pipe_distance, dist_cols),
                                  ('time', pipe_time, time_cols)]
                                  )

        # set final pipe
        self.pipeline = Pipeline(steps=[('feat_eng_bloc', feat_eng_bloc),
                                    ('regressor', self.estimator)])
    
        #return self

    def run(self):
        """set and train the pipeline"""
        self.pipeline.fit(self.X,self.y)

        #return self

    def evaluate(self, X_test, y_test):
        '''prints and returns the value of the RMSE'''
    
        y_pred = self.pipeline.predict(X_test)
        
        self.rmse = compute_rmse(y_pred, y_true=y_test)
        return self.rmse
    
    def save_model(self):
        self.mlflow_log_metric("rmse", self.rmse)
        self.mlflow_log_param("Model",self.estimator)
        joblib.dump(self,filepath+str(self.estimator))
        print("Model Saved")

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(EXPERIMENT_NAME)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)
    

if __name__ == "__main__":
    # store the data in a DataFrame
    df = get_data(nrows=10000)
    df = clean_data(df)

    # set X and y
    X = df.drop(columns='fare_amount')
    y = df.fare_amount

    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1)

    # build pipeline
    trainer = Trainer(X_train,y_train)
    trainer.set_pipeline()

    # train the pipeline
    trainer.run()

    # evaluate the pipeline
    print(trainer.evaluate(X_test, y_test))

    trainer.save_model()

