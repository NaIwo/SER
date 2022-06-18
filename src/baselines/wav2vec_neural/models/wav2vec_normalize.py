import tensorflow as tf


class Wav2vecNormalize(tf.keras.Model):
    def __init__(self, num_of_classes):
        super().__init__(name='wav2vec_classifier')

        hidden_size: int = 768
        self.linear1 = tf.keras.layers.Dense(hidden_size)
        self.linear2 = tf.keras.layers.Dense(hidden_size//4)
        self.linear3 = tf.keras.layers.Dense(hidden_size)
        self.mean = tf.keras.layers.Dense(hidden_size)
        self.std = tf.keras.layers.Dense(hidden_size)
        self.dropout = tf.keras.layers.Dropout(0.2)

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
        # Dev Loss 1.660027300255994, Accuracy: 0.3333333333333333 Average recall: 0.3125
        out = inputs
        for layer in [self.linear1, self.linear2, self.linear3]:
            out = layer(out, training=training)
            out = tf.nn.relu(out)
            out = self.dropout(out, training=training)
        mean = self.mean(out, training=training)
        std = tf.math.exp(self.std(out, training=training)) + 1e-10
        out = (inputs - mean) / std
        out = tf.nn.relu(out)
        return self.clf(out, training=training)
