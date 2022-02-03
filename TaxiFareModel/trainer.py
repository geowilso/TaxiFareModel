from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from sklearn.model_selection import train_test_split
import pandas as pd
from memoized_property import memoized_property
import mlflow
from mlflow.tracking import MlflowClient
import joblib

class Trainer():
    def __init__(self, X, y, estimator=LinearRegression()):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.MLFLOW_URI = "https://mlflow.lewagon.co/"
        self.experiment_name = "[UK] [London] [geowilso] TaxiFareModel 2"
        self.estimator = estimator

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([('dist_trans', DistanceTransformer()),
                              ('stdscaler', StandardScaler())])

        time_pipe = Pipeline([('time_enc', TimeFeaturesEncoder('pickup_datetime')),
                          ('ohe', OneHotEncoder(handle_unknown='ignore'))])
        preproc_pipe = ColumnTransformer([('distance', dist_pipe, [
        "pickup_latitude", "pickup_longitude", 'dropoff_latitude',
        'dropoff_longitude'
        ]), ('time', time_pipe, ['pickup_datetime'])],
                                     remainder="drop")
        self.pipeline = Pipeline([('preproc', preproc_pipe),
                                  ('linear_model', self.estimator)])


    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.mlflow_log_param("model", self.estimator)
        self.pipeline.fit(self.X, self.y)


    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        self.mlflow_log_metric("rmse", rmse)
        return rmse

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(self.MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(
                self.experiment_name).experiment_id

    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        ml_run_param = self.mlflow_run()
        self.mlflow_client.log_param(ml_run_param.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        ml_run_metric = self.mlflow_run()
        self.mlflow_client.log_metric(ml_run_metric.info.run_id, key, value)


    def save_model(self):
        """ Save the trained model into a model.joblib file """
        joblib.dump(self.pipeline, 'model.joblib')

if __name__ == "__main__":
    data = get_data()
    data = clean_data(data)
    y = data["fare_amount"]
    X = data.drop("fare_amount", axis=1)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
    #trainer = Trainer(X_train, y_train)
    #trainer.run()
    #rmse = trainer.evaluate(X_val, y_val)
    #print(rmse)

    for model in (LinearRegression(), Ridge(), Lasso(), ElasticNet(), SGDRegressor(), RandomForestRegressor()):
        trainer = Trainer(X_train, y_train, model)
        trainer.run()
        rmse = trainer.evaluate(X_val, y_val)
        print(rmse)

    experiment_id = trainer.mlflow_experiment_id
    print(f"experiment URL: https://mlflow.lewagon.co/#/experiments/{experiment_id}")
