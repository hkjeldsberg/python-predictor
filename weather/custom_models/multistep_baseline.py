import tensorflow as tf


class MultiStepLastBaseline(tf.keras.Model):
    def __init__(self, steps):
        super().__init__()
        self.steps = steps

    def call(self, inputs):
        return tf.tile(inputs[:, -1:, :], [1, self.steps, 1])
