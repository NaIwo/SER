import tensorflow as tf


class GemapsCNN(tf.keras.Model):
    def get_config(self):
        super(GemapsCNN, self).get_config()

    def __init__(self, num_of_classes: int, num_of_coefficients: int, num_of_windows: int, stride: int, filter_size: int):
        super().__init__(name='mfcc_classifier_cnn1d')

        self.clf = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(None, num_of_coefficients)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1D(256, filter_size, padding='same', activation='relu', strides=stride),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Conv1D(512, filter_size, padding='same', activation='relu', strides=stride),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.GlobalMaxPool1D(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(num_of_classes, activation='softmax')
        ])
        self.clf.summary()

    def call(self, inputs, training=None, mask=None):
        return self.clf(inputs, training=training)