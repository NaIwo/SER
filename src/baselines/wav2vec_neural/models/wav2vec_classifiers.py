import tensorflow as tf


class Wav2vecClassifier(tf.keras.Model):
    def __init__(self, num_of_classes):
        super().__init__(name='wav2vec_classifier')

        hidden_size: int = 768

        self.clf = tf.keras.models.Sequential([
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(hidden_size, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(hidden_size, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(hidden_size // 2, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(hidden_size // 2, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(hidden_size // 4, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(num_of_classes, activation='softmax')
        ])

    def call(self, inputs, training=None, mask=None):
        return self.clf(inputs, training=training)

