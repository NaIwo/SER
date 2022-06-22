from typing import Union, Tuple

import tensorflow as tf


class GemapsCNN(tf.keras.Model):
    def get_config(self):
        super(GemapsCNN, self).get_config()

    def __init__(self, num_of_classes: int, stride: int = 1, filter_size: int = 4):
        super().__init__(name='gemaps_classifier_cnn1d')

        self.clf = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(None, 25)),
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
        ], name="GeMAPS-Conv1D-network")
        self.clf.summary()

    def call(self, inputs, training=None, mask=None):
        return self.clf(inputs, training=training)


class StatsRegressor(tf.keras.Model):

    def __init__(self):
        super().__init__(name='stats_regressor')
        self.backbone = tf.keras.models.Sequential([
            #tf.keras.layers.InputLayer(input_shape=(None, num_of_coefficients)),
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
        ], name="mean-std-network")

        self.mean_layer = tf.keras.layers.Dense(25)
        self.std_layer = tf.keras.layers.Dense(25)

    def call(self, inputs: tf.Tensor, training=None, mask=None):
        hidden = self.backbone(tf.reshape(inputs[:, :300], shape=(tf.shape(inputs)[0], -1)))
        return self.mean_layer(hidden), tf.math.exp(self.std_layer(hidden)) + 1e-10


class GemapsCNNWithNormalization(tf.keras.Model):
    def __init__(self, num_of_classes):
        super().__init__(name='gemaps_classifier_cnn1d_with_actor_normalization')

        self.mean_std_regressor = StatsRegressor()

        self.num_of_classes = num_of_classes

        self.cnn = GemapsCNN(self.num_of_classes)

    def call(self, inputs: Union[Tuple[tf.Tensor, tf.Tensor], tf.Tensor], training=None, mask=None):
        features, statistics = inputs[0], inputs[1]
        predicted_means, predicted_stds = self.mean_std_regressor((features - tf.expand_dims(statistics[:, 0], axis=1)) / tf.expand_dims(statistics[:, 1], axis=1), training=training)
        if training:
            features = (features - tf.expand_dims(statistics[:, 0], axis=1)) / tf.expand_dims(statistics[:, 1], axis=1)
        else:
            features = (features - tf.expand_dims(predicted_means * 1000, axis=1)) / tf.expand_dims(predicted_stds * 1000, axis=1)
        return self.cnn.clf(features), predicted_means, predicted_stds


