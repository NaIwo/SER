import tensorflow as tf
from tensorflow import keras


class PaperNetwork(keras.Model):
    def __init__(self, num_of_classes: int):
        super().__init__(name='paper_network')
        self.gemaps_average = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2, padding='valid',
                                                               data_format='channels_last')
        self.gemaps_conv = tf.keras.layers.Conv1D(128, kernel_size=1, activation='relu')
        self.wav2vec_conv = tf.keras.layers.Conv1D(128, kernel_size=1, activation='relu')
        self.concat_conv = tf.keras.layers.Conv1D(128, kernel_size=1, activation='relu')
        self.wav2vec_layers_average = tf.keras.layers.Conv1D(1, kernel_size=1, use_bias=False,
                                                             kernel_initializer=tf.ones_initializer)
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.global_average = tf.keras.layers.GlobalAveragePooling1D()
        self.classifier = tf.keras.layers.Dense(num_of_classes, activation=None)

    def call(self, wav2vec: tf.Tensor, gemaps: tf.Tensor, training=None, mask=None, **kwargs):
        # wac2vec part
        wav2vec = tf.transpose(wav2vec, perm=[0, 2, 3, 1])
        f = self.wav2vec_layers_average(wav2vec)
        f = tf.squeeze(f, axis=-1)
        f /= tf.math.reduce_sum(self.wav2vec_layers_average.weights)
        wav_out = self.wav2vec_conv(f)
        wav_out = self.dropout(wav_out, training=training)
        # gemaps
        g_out = self.gemaps_average(gemaps)
        g_out = self.gemaps_conv(g_out)
        g_out = self.dropout(g_out, training=training)

        # concat
        concat = tf.keras.layers.concatenate([g_out, wav_out], axis=-1)
        concat = self.concat_conv(concat)
        concat = self.dropout(concat, training=training)
        global_average = self.global_average(concat)

        # clf
        results = self.classifier(global_average)
        return results

    def get_config(self):
        pass
