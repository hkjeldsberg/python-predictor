import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf


class WindowGenerator:
    def __init__(self, train_df, val_df, test_df, input_width, label_width, shift, label_columns=None):
        # Data
        self.test_df = test_df
        self.val_df = val_df
        self.train_df = train_df

        # Set label column indices
        self.label_columns = label_columns
        if self.label_columns is not None:
            self.label_columns_indicies = {name: i for i, name in enumerate(self.label_columns)}

        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        # Window params
        self.shift = shift
        self.label_width = label_width
        self.input_width = input_width
        self.total_window_size = self.input_width + shift

        self.input_slice = slice(0, self.input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_inputs_and_labels(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1
            )

        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot(self, model=None, plot_col='T (degC)', inputs=None, labels=None):
        plot_col_index = self.column_indices[plot_col]
        n = len(inputs.shape)

        for i in range(n):
            plt.subplot(n, 1, i + 1)

            # Plot inputs
            plt.plot(self.input_indices, inputs[i, :, plot_col_index], marker=".", label="Input")

            # Plot labels
            if self.label_columns:
                label_col_index = self.label_columns_indicies.get(plot_col, None)
            else:
                label_col_index = plot_col_index
            plt.scatter(self.label_indices, labels[i, :, label_col_index],
                        edgecolors="black", label="Labels", color="red", s=64)

            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[i, :, label_col_index],
                            marker="X", label="Predictions", edgecolor="black", color="green", s=64)

            if i == 0:
                plt.legend()
            plt.xlabel("Time [h]")
            plt.ylabel(f"{plot_col} [normalized]")

        plt.show()

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32
        )

        ds = ds.map(self.split_inputs_and_labels)

        return ds
