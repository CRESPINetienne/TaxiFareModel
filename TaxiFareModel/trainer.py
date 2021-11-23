from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data
from sklearn.model_selection import train_test_split
import numpy as np


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

        dist_pipe = Pipeline([
    ('dist_trans', DistanceTransformer()),
    ('stdscaler', StandardScaler())
    ])

        time_pipe = Pipeline([
    ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ])
        preproc_pipe = ColumnTransformer([
    ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
    ('time', time_pipe, ['pickup_datetime'])
    ], remainder="drop")

        self.pipeline = Pipeline([
    ('preproc', preproc_pipe),
    ('linear_model', LinearRegression())
    ])
        return self.pipeline

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)
        return self.pipeline

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        return np.sqrt(((y_pred - y_test)**2).mean())


if __name__ == "__main__":
    df = get_data()
    df =clean_data(df)
    y = df.pop("fare_amount")
    X = df
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    pipeline_test = Trainer(X_train, y_train)
    pipeline_test.set_pipeline()
    pipeline_test.run()
    print(pipeline_test.evaluate(X_train, y_test))
