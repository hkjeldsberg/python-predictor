import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
from IPython import embed
from keras.src.activations import linear
from matplotlib import pyplot as plt
from os import path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from baseline import Baseline
from window_generator import WindowGenerator

sns.set_theme()


class Forecasting:
    def __init__(self, config):
        self.config = config
        self.results_dir = self.config['results_dir']
        self.loss = self.config["loss"]
        self.optimizer = self.config["optimizer"]
        self.metrics = self.config["metrics"]
        self.units = self.config["units"]
        self.activation = self.config["activation"]
        self.epochs = self.config["epochs"]

        # Define performance dicts
        self.val_performance = {}
        self.performance = {}

    def read_data(self):
        df = pd.read_csv(path.join(self.results_dir, "weather_data.csv"))
        df = df[5::6]

        self.df = df

    def create_features(self):
        # Create wind vector
        wv = self.df.pop('wv (m/s)')
        wd_rad = self.df.pop('wd (deg)') * np.pi / 180

        self.df['Wx'] = wv * np.cos(wd_rad)
        self.df['Wy'] = wv * np.sin(wd_rad)

        # Create h
        date_time = pd.to_datetime(self.df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
        timestamp_s = date_time.map(pd.Timestamp.timestamp)
        day = 24 * 60 * 60
        year = 365.2425 * day

        self.df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
        self.df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
        self.df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
        self.df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

    def preprocess_data(self):
        self.df.drop(columns=[
            'SWDR (W/m**2)', 'SDUR (s)', 'TRAD (degC)', 'Rn (W/m**2)', 'ST002 (degC)', 'ST004 (degC)',
            'ST008 (degC)', 'ST016 (degC)', 'ST032 (degC)', 'ST064 (degC)', 'ST128 (degC)', 'SM008 (%)', 'SM016 (%)',
            'SM032 (%)', 'SM064 (%)', 'SM128 (%)', 'rain (mm)'
        ], inplace=True)
        # Remove outliers
        columns = ['p (mbar)', 'rho (g/m**3)']
        for col in columns:
            self.df = self.df[(self.df[col] > 0)]

        # Set minimum values of -9999 to 0
        for col in self.df.columns:
            min_value = self.df[col] == -9999.0
            self.df.loc[min_value, col] = 0.0

        self.create_features()
        self.split_data()

    def plot_data(self):
        cols = ['T (degC)', 'p (mbar)', 'rho (g/m**3)']
        features = self.df[cols]
        features.plot(subplots=True)
        plt.show()

        features = self.df[cols][:500]
        features.plot(subplots=True)
        plt.show()

        plt.hist2d(self.df['Wx'], self.df['Wy'], bins=(100, 100), vmax=400)
        plt.colorbar()
        plt.xlabel('Wind x [m/s]')
        plt.ylabel('Wind y [m/s]')
        plt.show()

        df_std = (self.train_df - self.train_df.mean()) / self.train_df.std()
        df_std = df_std.melt(var_name='Column', value_name='Normalized')
        plt.figure(figsize=(12, 6))
        ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
        ax.set_xticklabels(self.df.keys(), rotation=90)
        plt.show()

    def split_data(self):
        # Split
        n = len(self.df)
        train_df = self.df[0:int(n * 0.7)]
        val_df = self.df[int(n * 0.7):int(n * 0.9)]
        test_df = self.df[int(n * 0.9):]

        # Normalization
        train_mean = train_df.mean()
        train_std = train_df.std()

        self.train_df = (train_df - train_mean) / train_std
        self.val_df = (val_df - train_mean) / train_std
        self.test_df = (test_df - train_mean) / train_std

    def create_wide_window(self):
        self.wide_window = WindowGenerator(
            train_df=self.train_df, test_df=self.test_df, val_df=self.val_df,
            input_width=24, label_width=24, shift=1,
            label_columns=['T (degC)']
        )
        print(self.wide_window)

    def create_single_step_window(self):
        self.single_step_window = WindowGenerator(
            train_df=self.train_df, test_df=self.test_df, val_df=self.val_df,
            input_width=1, label_width=1, shift=1,
            label_columns=['T (degC)']
        )
        print(self.single_step_window)

    def plot_single_step(self):
        for inputs, labels in self.single_step_window.train.take(1):
            print(f'Inputs shape (batch, time, features): {inputs.shape}')
            print(f'Labels shape (batch, time, features): {labels.shape}')
            self.single_step_window.plot(inputs=inputs, labels=labels)

    def train_baseline(self):
        window = self.wide_window
        for inputs, labels in window.train.take(1):
            print(f'Inputs shape (batch, time, features): {inputs.shape}')
            print(f'Labels shape (batch, time, features): {labels.shape}')

        model = Baseline(label_index=window.column_indices['T (degC)'])

        history = self.compile_and_fit(model, window)

        self.val_performance['Baseline'] = model.evaluate(window.val, return_dict=True)
        self.performance['Baseline'] = model.evaluate(window.test, verbose=0, return_dict=True)

        window.plot(model, inputs=inputs, labels=labels)

    def train_linear(self):
        window = self.wide_window
        for inputs, labels in window.train.take(1):
            print(f'Inputs shape (batch, time, features): {inputs.shape}')
            print(f'Labels shape (batch, time, features): {labels.shape}')
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=self.units),
        ])
        print('Input shape:', window.example[0].shape)
        print('Output shape:', linear(window.example[0]).shape)

        history = self.compile_and_fit(model, self.single_step_window)

        self.val_performance['Linear'] = model.evaluate(window.val, return_dict=True)
        self.performance['Linear'] = model.evaluate(window.test, verbose=0, return_dict=True)

        window.plot(model, inputs=inputs, labels=labels)

    def compile_and_fit(self, model, window, patience=2):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            mode="min"
        )
        model.compile(
            loss=self.loss,
            optimizer=self.optimizer,
            metrics=self.metrics
        )
        history = model.fit(
            window.train,
            epochs=self.epochs,
            validation_data=window.val,
            callbacks=[early_stopping]
        )

        return history


def main(config):
    model = Forecasting(config)
    model.read_data()
    model.preprocess_data()
    model.create_single_step_window()
    model.create_wide_window()
    model.train_baseline()


if __name__ == '__main__':
    config = {
        "results_dir": "data",
        "loss": tf.keras.losses.MeanSquaredError(),
        "optimizer": tf.keras.optimizers.Adam(),
        "metrics": [tf.keras.metrics.MeanAbsoluteError()],
        "activation": "relu",
        "units": 1,
        "epochs": 20
    }
    main(config)