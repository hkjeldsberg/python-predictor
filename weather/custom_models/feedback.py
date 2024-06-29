import tensorflow as tf


class Feedback(tf.keras.Model):
    def __init__(self, units, out_steps, num_features):
        super().__init__()
        self.num_features = num_features
        self.units = units
        self.out_steps = out_steps
        self.lstm_cell = tf.keras.layers.LSTMCell(self.units)
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(self.num_features)

    def warmup(self, inputs):
        x, *state = self.lstm_rnn(inputs)

        predictions = self.dense(x)
        return predictions, state

    def call(self, inputs,  training=None):
        predictions = []
        prediction, state = self.warmup(inputs)
        predictions.append(prediction)

        for i in range(1, self.out_steps):
            x = prediction
            x, state = self.lstm_cell(x, states=state, training=training)
            prediction = self.dense(x)
            predictions.append(prediction)

        predictions = tf.stack(predictions)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions
