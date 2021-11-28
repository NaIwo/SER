import tensorflow as tf


class MfccCNN(tf.keras.Model):  # worked well on MFCCs / RAVDESS
    def get_config(self):
        super(MfccCNN, self).get_config()

    def __init__(self, num_of_classes: int, num_of_coefficients: int, num_of_windows: int):
        super().__init__(name='mfcc_classifier_cnn2d')

        self.clf = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(num_of_coefficients, num_of_windows, 1)),
            tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', strides=(2, 1)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.AvgPool2D(pool_size=(249, 1)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_of_classes, activation='softmax')
        ])

    def call(self, inputs, training=None, mask=None):
        return self.clf(inputs, training=training)
