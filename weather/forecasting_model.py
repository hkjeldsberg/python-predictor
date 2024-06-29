import argparse
import json

import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
from IPython import embed
from matplotlib import pyplot as plt
from os import path

from baseline import Baseline
from residual_wrapper import ResidualWrapper
from multistep_baseline import MultiStepLastBaseline
from repeat_baseline import RepeatBaseline
from feedback import Feedback
from window_generator import WindowGenerator

sns.set_theme()
sns.color_palette("husl", 8)


class Forecasting:
    def __init__(self, config):
        self.config = config
        self.num_features = self.config['num_features']
        self.data_dir = self.config['data_dir']
        self.results_dir = self.config['results_dir']
        self.loss = self.config["loss"]
        self.optimizer = self.config["optimizer"]
        self.metrics = self.config["metrics"]
        self.units = self.config["units"]
        self.units_dense = self.config["units_dense"]
        self.activation = self.config["activation"]
        self.epochs = self.config["epochs"]
        self.conv_width = self.config['conv_width']
        self.label_width = self.config['label_width']
        self.shift = self.config['shift']
        self.input_width = self.config['input_width']

    def read_data(self):
        df = pd.read_csv(path.join(self.data_dir, "weather_data.csv"))
        df = df[5::6]

        self.df = df

        # Define performance dicts
        with open(path.join(self.results_dir, "performance_multi.json")) as f:
            self.performance = json.loads(f.read())
        with open(path.join(self.results_dir, "val_performance_multi.json")) as f:
            self.val_performance = json.loads(f.read())

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

    def create_conv_window(self):
        self.conv_window = WindowGenerator(
            train_df=self.train_df, test_df=self.test_df, val_df=self.val_df,
            input_width=self.conv_width, label_width=self.label_width, shift=1,
            label_columns=['T (degC)']
        )
        print(self.conv_window)

    def create_wide_conv_window(self):
        input_width = self.label_width + (self.conv_width - 1)

        self.wide_conv_window = WindowGenerator(
            train_df=self.train_df, test_df=self.test_df, val_df=self.val_df,
            input_width=input_width, label_width=self.label_width, shift=1,
            label_columns=['T (degC)']
        )
        print(self.wide_conv_window)

    def create_multi_window(self):
        self.multi_window = WindowGenerator(
            train_df=self.train_df, test_df=self.test_df, val_df=self.val_df,
            input_width=self.input_width, label_width=self.shift, shift=self.shift,
        )
        print(self.multi_window)

    def train_multibaseline(self):
        window = self.multi_window
        model = MultiStepLastBaseline(
            steps=self.shift
        )

        self.train(model, window, "LastBaseline")

    def train_repeatbaseline(self):
        window = self.multi_window
        model = RepeatBaseline()

        self.train(model, window, "RepeatBaseline")

    def train_multi_linear(self):
        window = self.multi_window
        model = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
            tf.keras.layers.Dense(
                units=self.shift * self.num_features,
                kernel_initializer=tf.keras.initializers.zeros(),
            ),
            tf.keras.layers.Reshape([self.shift, self.num_features])
        ])

        self.train(model, window, "MultiLinear")

    def train_multi_dense(self):
        window = self.multi_window
        model = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
            tf.keras.layers.Dense(units=512, activation=self.activation),
            tf.keras.layers.Dense(
                units=self.shift * self.num_features,
                kernel_initializer=tf.keras.initializers.zeros(),
            ),
            tf.keras.layers.Reshape([self.shift, self.num_features])
        ])

        self.train(model, window, "MultiDense")

    def train_multi_conv(self):
        window = self.multi_window
        model = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: x[:, -self.conv_width:, :]),
            tf.keras.layers.Conv1D(256, activation='relu', kernel_size=self.conv_width),
            tf.keras.layers.Dense(
                self.shift * self.num_features,
                kernel_initializer=tf.initializers.zeros()
            ),
            tf.keras.layers.Reshape([self.shift, self.num_features])
        ])
        self.train(model, window, "MultiConv")

    def train_multi_rnn(self):
        window = self.multi_window
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(
                units=self.units,
                return_sequences=False,
            ),
            tf.keras.layers.Dense(
                self.shift * self.num_features,
                kernel_initializer=tf.keras.initializers.zeros()
            ),
            tf.keras.layers.Reshape([self.shift, self.num_features])
        ])

        self.train(model, window, "MultiRNN")

    def train_autoregressive_rnn(self):
        window = self.multi_window
        model = Feedback(units=self.units, out_steps=self.shift, num_features=self.num_features)
        prediction, state = model.warmup(window.example[0])
        print("Prediction (shape):", prediction.shape)

        self.train(model, window, "AR LSTM")

    def train_baseline(self):
        window = self.wide_window
        model = Baseline(
            label_index=window.column_indices['T (degC)']
        )
        self.train(model, window, "Baaseline")

    def train_linear(self):
        window = self.wide_window
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=1),
        ])
        self.train(model, window, "Linear")

    def train_dense(self):
        window = self.single_step_window
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=self.units_dense, activation=self.activation),
            tf.keras.layers.Dense(units=self.units_dense, activation=self.activation),
            tf.keras.layers.Dense(units=self.num_features)
        ])
        self.train(model, window, "Dense")

    def train_rnn(self):
        window = self.wide_window
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(units=self.units, return_sequences=True),
            tf.keras.layers.Dense(units=self.num_features)
        ])
        self.print_model_info(model, window)
        self.train(model, window, "LSTM")

    def train_multi_step_dense(self):
        window = self.wide_conv_window
        model = tf.keras.Sequential([
            # (Time, Features) -> (Time*Features)
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=self.units, activation=self.activation),
            tf.keras.layers.Dense(units=self.units, activation=self.activation),
            tf.keras.layers.Dense(units=1),
            # Reshape
            tf.keras.layers.Reshape([1, -1])
        ])

        self.train(model, window, "Multi step dense")

    def train_conv(self):
        window = self.wide_conv_window

        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(
                filters=32,
                kernel_size=(self.conv_width,),
                activation=self.activation,
            ),
            tf.keras.layers.Dense(units=self.units, activation=self.activation),
            tf.keras.layers.Dense(units=1),
        ])
        self.print_model_info(model, window)
        self.train(model, window, "Conv")

    def train_residual_lstm(self):
        window = self.wide_window
        model = ResidualWrapper(
            tf.keras.Sequential([
                tf.keras.layers.LSTM(units=self.units, return_sequences=True),
                tf.keras.layers.Dense(
                    units=self.num_features,
                    kernel_initializer=tf.initializers.zeros()
                )
            ])
        )

        self.print_model_info(model, window)
        self.train(model, window, "Residual LSTM")

    def compile_and_fit(self, model, window, patience=2):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            mode="min"
        )
        model.compile(
            run_eagerly=True,
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

    def evaluate_model(self, model, window, name, plot=False):
        self.val_performance[name] = model.evaluate(window.val, return_dict=True)
        self.performance[name] = model.evaluate(window.test, verbose=0, return_dict=True)
        if plot:
            window.plot(model)

    def train(self, model, window, name):
        history = self.compile_and_fit(model, window)
        self.evaluate_model(model, window, name, True)
        self.save_error()

    def print_model_info(self, model, window):
        print("Wide window")
        print('Input shape:', window.example[0].shape)
        print('Labels shape:', window.example[1].shape)
        print('Output shape:', model(window.example[0]).shape)

    def plot_error(self):
        x = np.arange(len(self.performance))
        width = 0.3
        metric_name = 'mean_absolute_error'
        val_mae = [v[metric_name] for v in self.val_performance.values()]
        test_mae = [v[metric_name] for v in self.performance.values()]

        # plt.ylabel('Mean absolute error – Normalized Temperature (Celcius)')
        plt.ylabel("Mean absolute error – Averaged over all features")
        plt.bar(x - 0.17, val_mae, width, label='Validation', color='firebrick')
        plt.bar(x + 0.17, test_mae, width, label='Test', color="darkgreen")
        plt.xticks(ticks=x, labels=self.performance.keys(),
                   rotation=45)
        plt.legend()
        plt.show()

    def save_error(self):
        with open(path.join(self.results_dir, "performance_all.json"), "w") as f:
            f.write(json.dumps(self.performance))
        with open(path.join(self.results_dir, "val_performance_all.json"), "w") as f:
            f.write(json.dumps(self.val_performance))


def main(config, window_type, model_name):
    model = Forecasting(config)
    model.read_data()
    model.preprocess_data()

    if window_type == "single":
        model.create_single_step_window()
        model.create_wide_window()
        model.create_conv_window()
        model.create_wide_conv_window()
    if window_type == "multi":
        model.create_multi_window()

    models_single = {
        "baseline": model.train_baseline,
        "linear": model.train_linear,
        "dense": model.train_dense,
        "multi_step": model.train_multi_step_dense,
        "conv": model.train_conv,
        "rnn": model.train_rnn,
        "res_rnn": model.train_residual_lstm
    }

    models_multi = {
        "baseline": model.train_multibaseline,
        "repeat": model.train_repeatbaseline,
        "dense": model.train_multi_dense,
        "conv": model.train_multi_conv,
        "rnn": model.train_multi_rnn,
        "ar_rnn": model.train_autoregressive_rnn
    }

    if window_type == "single" and model_name in models_single:
        models_single[model_name]()
    elif window_type == "multi" and model_name in models_multi:
        models_multi[model_name]()
    else:
        print(f"Model {model_name} not found for window type {window_type}.")

    model.plot_error()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a forecasting model.")
    parser.add_argument('--window_type', type=str, required=True,
                        choices=['single', 'multi'],
                        help='Type of window to use for training the model.')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name of the model to train.')

    args = parser.parse_args()

    config = {
        "data_dir": "data",
        "results_dir": "results",
        "loss": tf.keras.losses.MeanSquaredError(),
        "optimizer": tf.keras.optimizers.Adam(),
        "metrics": [tf.keras.metrics.MeanAbsoluteError()],
        "activation": "relu",
        "units": 32,
        "units_dense": 64,
        "epochs": 20,
        "conv_width": 10,
        "label_width": 48,
        "num_features": 17,
        "shift": 24,
        "input_width": 24
    }

    main(config, args.window_type, args.model_name)
