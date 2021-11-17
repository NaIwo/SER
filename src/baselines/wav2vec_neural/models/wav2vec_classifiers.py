import tensorflow as tf

from src.preprocessors.wav2vec import Wav2VecModel


class Wav2vecClassifier(tf.keras.Model):
    def __init__(self, num_of_classes):
        super().__init__(name='wav2vec_classifier')

        self.wav2vec = Wav2VecModel(num_of_classes=num_of_classes)

        self.clf = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.wav2vec.config.hidden_size,)),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(self.wav2vec.config.hidden_size, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(self.wav2vec.config.hidden_size, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(self.wav2vec.config.hidden_size // 2, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(self.wav2vec.config.hidden_size // 4, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(num_of_classes, activation='softmax')
        ])

    def call(self, inputs, training=None, mask=None):
        return self.clf(inputs, training=training)


class Wav2vecClassifierCNN(tf.keras.Model):
    def __init__(self, num_of_classes):
        super().__init__(name='wav2vec_classifier_cnn')

        self.wav2vec = Wav2VecModel(num_of_classes=num_of_classes)

        self.clf = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(176, self.wav2vec.config.hidden_size)),  # 176 - change it
            tf.keras.layers.Conv1D(256, 3, padding='same', activation='relu', strides=2),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Conv1D(256, 3, padding='same', activation='relu', strides=2),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.GlobalAvgPool1D(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(num_of_classes, activation='softmax')
        ])

    def call(self, inputs, training=None, mask=None):
        return self.clf(inputs, training=training)
