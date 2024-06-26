from os import path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR

pd.set_option("future.no_silent_downcasting", True)
sns.set_theme()


class CovidForecaster:
    def __init__(self, config):
        self.config = config
        self.alpha = config['alpha']
        self.lag = config['lag']
        self.horizon = config['horizon']
        self.data_dir = self.config['data_dir']

        self.k_features = self.config['k_features']
        self.use_normalization = self.config['use_normalization']

    def read_data(self, print_info=False):
        try:
            df = pd.read_csv(path.join(self.data_dir, "covid_de.csv"))
            df['date'] = pd.to_datetime(df.date, format='%Y-%m-%d')
            df = df.groupby(['date']).sum()[['cases', 'deaths']]
            df.to_csv("data/covid_de_update.csv")
        except:
            df = pd.read_csv(path.join(self.data_dir, "covid_de_update.csv"))

        self.df = df

    def preprocess_data(self):
        self.split_dataset()
        self.check_if_stationary(self.df_diff)
        self.scale_dataset()
        self.scale_test_dataset()
        self.granger_causality_test()

    def split_dataset(self):
        self.df_roll = self.df.rolling(7).mean().dropna()
        cutoff_index = int(self.df_roll.shape[0] * 0.9)
        self.X_train = self.df_roll.iloc[:cutoff_index]
        self.X_test = self.df_roll.iloc[cutoff_index:]
        self.df_diff = self.X_train.diff().dropna()
        self.df_diff_test = self.X_test.diff().dropna()

    def check_if_stationary(self, df):
        for var in df.columns:
            result = adfuller(df[var])

            p_value = result[1]
            print(f"p-value: {p_value:.5f}")
            if p_value <= self.alpha:
                print(f"The variable {var} is stationary.\n")
            else:
                print(f"The variable {var} is NOT stationary.\n")

    def scale_dataset(self):
        self.scaler = StandardScaler()
        scaled_values = self.scaler.fit_transform(self.df_diff)

        self.df_scaled = pd.DataFrame(
            scaled_values,
            columns=self.df_diff.columns,
            index=self.df_diff.index
        )

    def scale_test_dataset(self):
        scaled_values = self.scaler.fit_transform(self.df_diff_test)

        df_scaled = pd.DataFrame(
            scaled_values,
            columns=self.df_diff_test.columns,
            index=self.df_diff_test.index
        )

        self.df_test_processed = df_scaled[df_scaled.index > self.df_scaled.index[-1]]

    def inverse_transformation(self, df_processed, df, scaler):
        df_diff = pd.DataFrame(
            scaler.inverse_transform(df_processed),
            columns=df_processed.columns,
            index=df_processed.index
        )
        self.df_original = df_diff.cumsum() + df[df.index < df_diff.index[0]].iloc[-1]

    def granger_causality_test(self):
        deaths_as_cause = grangercausalitytests(self.df_scaled[['cases', 'deaths']], self.lag)
        cases_as_cause = grangercausalitytests(self.df_scaled[['deaths', 'cases']], self.lag)

    def train(self, model="VAR"):
        if model == "VAR":
            self.model = VAR(self.df_scaled)
        else:
            print(f"{model} is not a valid model.")
            exit()

        optimal_lags = self.model.select_order()
        print(f"Optimal lag orders: {optimal_lags.selected_orders}")

        # Use Bayesian informaton criterion (BIC)
        self.lag_order = optimal_lags.selected_orders['bic']
        self.results = self.model.fit(self.lag_order)

    def predict(self):
        var_model = self.results.model
        print(self.results.summary())
        forecast = self.results.forecast(self.df_scaled.values[-self.lag_order:], steps=self.horizon)
        self.y_pred = pd.DataFrame(
            forecast,
            columns=self.df_scaled.columns,
            index=self.X_test.iloc[:self.horizon].index
        )

    def plot_data(self):
        sns.lineplot(self.X_train, palette=['b'])
        sns.lineplot(self.X_test, palette=['r'])
        plt.xlabel("Date")
        plt.ylabel("Number of cases/deaths")
        plt.xticks(rotation=45)
        plt.show()

    def plot_diff(self):
        sns.lineplot(self.X_train.diff().dropna(), palette=['b'])
        sns.lineplot(self.X_test.diff().dropna(), palette=['r'])
        plt.xlabel("Date")
        plt.ylabel("Number of cases/deaths")
        plt.xticks(rotation=45)
        plt.show()

    def plot_prediction(self):
        self.inverse_transformation(self.y_pred, self.df_roll, self.scaler)
        fig, (ax1, ax2) = plt.subplots(2, 1)
        self.X_train[-30:].cases.plot(ax=ax1)
        self.X_test[:self.horizon].cases.plot(ax=ax1)
        self.df_original.cases.plot(ax=ax1)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Increment in number of cases')
        ax1.legend(['Train', 'Test', 'Forecast'])

        self.X_train[-30:].deaths.plot(ax=ax2)
        self.X_test[:self.horizon].deaths.plot(ax=ax2)
        self.df_original.deaths.plot(ax=ax2)
        ax2.grid(alpha=0.5, which='both')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Increment in number of deaths')
        ax2.legend(['Train', 'Test', 'Forecast'])
        plt.show()

    def plot_prediction_scaled(self):
        fig, (ax1, ax2) = plt.subplots(2, 1)
        self.df_scaled[-30:].cases.plot(ax=ax1)
        self.df_test_processed[:self.horizon].cases.plot(ax=ax1)
        self.y_pred.cases.plot(ax=ax1)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Increment in number of cases')
        ax1.legend(['Train', 'Test', 'Forecast'])

        self.df_scaled[-30:].deaths.plot(ax=ax2)
        self.df_test_processed[:self.horizon].deaths.plot(ax=ax2)
        self.y_pred.deaths.plot(ax=ax2)
        ax1.grid(alpha=0.5, which='both')
        ax2.grid(alpha=0.5, which='both')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Increment in number of deaths')
        ax2.legend(['Train', 'Test', 'Forecast'])
        plt.show()


def main(config):
    model = CovidForecaster(config)
    model.read_data()
    model.preprocess_data()
    model.train()
    model.predict()
    model.plot_prediction()


if __name__ == '__main__':
    config = {
        'alpha': 0.05,  # For checking stationary data
        'lag': 5,  # Granger causality test lag
        'horizon': 14,
        'data_dir': 'data',
        'k_features': 4,
        'use_normalization': False
    }
    main(config)
