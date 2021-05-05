# imports
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

from TaxiFareModel.encoders import DistanceTransformer,TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data,clean_data

import pandas as pd


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

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
                                    ('regressor', KNeighborsRegressor())])
    
        return self

    def run(self):
        """set and train the pipeline"""
        self.pipeline.fit(self.X,self.y)

        return self

    def evaluate(self, X_test, y_test ):
        '''prints and returns the value of the RMSE'''
    
        y_pred = self.pipeline.predict(X_test)
        
        rmse = compute_rmse(y_pred, y_true=y_test)
        return rmse


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
    pipe = trainer.run()

    # evaluate the pipeline
    aa = trainer.evaluate(X_test, y_test, pipe)
    print(aa)
