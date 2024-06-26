from os import path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

pd.set_option("future.no_silent_downcasting", True)
sns.set_theme()


class BTCForecaster:
    def __init__(self, config):
        self.config = config

        self.results_dir = self.config['results_dir']
        self.data_dir = self.config['data_dir']
        self.test_size = self.config['test_size']
        self.k_features = self.config['k_features']
        self.use_normalization = self.config['use_normalization']

    def read_data(self, print_info=False):
        df = pd.read_csv(path.join(self.data_dir, "BTC-USD.csv"))
        df = df[['Date', 'Close']]
        df.index = pd.to_datetime(df['Date'], format="%Y-%m-%d")
        df.drop(columns=['Date'])

        self.df = df

    def preprocess_data(self):
        self.split_dataset()

    def split_dataset(self):
        self.X_train = self.df[self.df.index < pd.to_datetime("2020-11-01", format='%Y-%m-%d')]
        self.X_test = self.df[self.df.index > pd.to_datetime("2020-11-01", format='%Y-%m-%d')]
        self.y_train = self.X_train['Close']
        self.y_test = self.X_test['Close']

    def train(self, model="ARMA"):
        if model == "ARMA":
            self.model = SARIMAX(self.y_train, order=(1, 0, 1))
        elif model == "ARIMA":
            self.model = ARIMA(self.y_train, order=(5, 4, 2))
        elif model == "SARIMA":
            self.model = SARIMAX(self.y_train, order=(5, 4, 2), seasonal_order=(2, 2, 2, 12))
        else:
            print(f"{model} is not a valid model.")
            exit()

        self.results = self.model.fit()

    def predict(self):
        self.y_pred = self.results.get_forecast(steps=self.X_test.shape[0])
        self.y_pred_df = self.y_pred.conf_int(alpha=0.05)
        self.y_pred_df['Predictions'] = self.results.predict(start=self.y_pred_df.index[0],
                                                             end=self.y_pred_df.index[-1])
        self.y_pred_df.index = self.X_test.index
        self.y_pred = self.y_pred_df['Predictions']

    def validate_model(self, model):
        # Common model performance metrics
        rmse = np.sqrt(mean_squared_error(self.y_test.values, self.y_pred.values))

        return rmse

    def roc_curve(self, model="LR", show_plot=False):
        # Plot Reciever operating characteristic (ROC) curve
        y_pred_prob = self.model.predict_proba(self.X_test)[::, 1]
        false_pos, true_pos, _ = metrics.roc_curve(self.y_test, y_pred_prob)

        auc = metrics.roc_auc_score(self.y_test, y_pred_prob)
        if show_plot:
            plt.plot(false_pos, true_pos, label=f"Model={model}, AUC={auc:.3f}")
            plt.title('ROC Curve')
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.legend()
            plt.show()

    def plot_data(self):
        sns.lineplot(self.X_train, palette=['b'])
        sns.lineplot(self.X_test, palette=['r'])
        sns.lineplot(self.y_pred_df['Predictions'], palette=['g'])
        plt.xlabel("Date")
        plt.ylabel("Closing price")
        plt.xticks(rotation=45)
        plt.show()


def main(config):
    models = ['ARMA', 'ARIMA', "SARIMA"]
    rmses = []
    model = BTCForecaster(config)
    model.read_data()
    model.preprocess_data()

    for model_name in models:
        model.train(model_name)
        model.predict()
        model.plot_data()
        rmse = model.validate_model(model_name)
        rmses.append(rmse)

    print(models)
    print(rmses)


if __name__ == '__main__':
    config = {
        'test_size': 0.4,  # [0,1]
        'results_dir': 'results',
        'data_dir': 'data',
        'k_features': 4,
        'use_normalization': False
    }
    main(config)
