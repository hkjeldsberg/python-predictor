import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
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

    def read_data(self):
        df = pd.read_csv(path.join(self.results_dir, "weather_data.csv"))
        df = df[5::6]
        df.index = pd.to_datetime(df['Date Time'], format="%d.%m.%Y %H:%M:%S")
        df.dropna()

        self.df = df

    def create_features(self):
        # Create wind vector
        wv = self.df.pop('wv (m/s)')
        wd_rad = self.df.pop('wd (deg)') * np.pi / 180

        self.df['Wx'] = wv * np.cos(wd_rad)
        self.df['Wy'] = wv * np.sin(wd_rad)

    def preprocess_data(self):
        self.df.drop(columns=[
            'Date Time', 'SWDR (W/m**2)', 'SDUR (s)', 'TRAD (degC)', 'Rn (W/m**2)', 'ST002 (degC)', 'ST004 (degC)',
            'ST008 (degC)', 'ST016 (degC)', 'ST032 (degC)', 'ST064 (degC)', 'ST128 (degC)', 'SM008 (%)', 'SM016 (%)',
            'SM032 (%)', 'SM064 (%)', 'SM128 (%)'
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
        train_df, temp_df = train_test_split(self.df, test_size=0.3, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=1 / 3, random_state=42)

        # Normalization
        scaler = StandardScaler()
        self.train_df = scaler.fit_transform(train_df)
        self.val_df = scaler.transform(val_df)
        self.test_df = scaler.transform(test_df)

        self.test_df = pd.DataFrame(columns=test_df.columns, data=self.test_df)
        self.train_df = pd.DataFrame(columns=train_df.columns, data=self.train_df)
        self.val_df = pd.DataFrame(columns=val_df.columns, data=self.val_df)

    def create_wide_window(self):
        self.wide_window = WindowGenerator(train_df=self.train_df, test_df=self.test_df, val_df=self.val_df,
                                           input_width=24, label_width=24, shift=1, label_columns=['T (degC)'])
        print(self.wide_window)

    def create_single_step_window(self):
        self.single_step_window = WindowGenerator(train_df=self.train_df, test_df=self.test_df, val_df=self.val_df,
                                                  input_width=1, label_width=1, shift=1, label_columns=['T (degC)'])
        print(self.single_step_window)

    def train_baseline(self):
        for inputs, labels in self.wide_window.train.take(1):
            print(f'Inputs shape (batch, time, features): {inputs.shape}')
            print(f'Labels shape (batch, time, features): {labels.shape}')
            self.wide_window.plot(inputs=inputs, labels=labels)

        model = Baseline(label_index=self.wide_window.column_indices['T (degC)'])
        model.compile(
            loss=config['loss'],
            metrics=config['metrics']
        )
        val_performance = {}
        performance = {}
        val_performance['Baseline'] = model.evaluate(self.wide_window.val, return_dict=True)
        performance['Baseline'] = model.evaluate(self.wide_window.test, verbose=0, return_dict=True)

        self.wide_window.plot(model, inputs=inputs, labels=labels)

    def train_linear(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=self.units, activation=self.activation),
        ])
        print('Input shape:', self.single_step_window.example[0].shape)
        print('Output shape:', linear(self.single_step_window.example[0]).shape)


def main(config):
    model = Forecasting(config)
    model.read_data()
    model.preprocess_data()
    model.create_single_step_window()
    model.train_linear()


if __name__ == '__main__':
    config = {
        "results_dir": "data",
        "loss": tf.keras.losses.MeanSquaredError(),
        "optimizer": tf.keras.optimizers.Adam(),
        "metrics": [tf.keras.metrics.MeanAbsoluteError()],
        "activation": "relu",
        "units": 1
    }
    main(config)
