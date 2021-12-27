import tensorflow as tf
import numpy as np
from sklearn.metrics import recall_score, accuracy_score
import os
from pathlib import Path


gemaps_normalizer = tf.keras.layers.Normalization(axis=-1)
wav2vec_normalizer = tf.keras.layers.Normalization(axis=-1)

def train(model: tf.keras.Model, train_ds_wav2vec: tf.data.Dataset, val_ds_wav2vec: tf.data.Dataset,
          train_ds_gemaps: tf.data.Dataset, val_ds_gemaps: tf.data.Dataset, epochs: int):

    global wav2vec_normalizer, gemaps_normalizer
    for step, ((x_wav, y_wav), (x_ge, y_ge)) in enumerate(zip(train_ds_wav2vec, train_ds_gemaps)):
        if step == 0:
            wav2vec_normalizer.adapt(x_wav)
            gemaps_normalizer.adapt(x_ge)
        else:
            wav2vec_normalizer.update_state(x_wav)
            gemaps_normalizer.update_state(x_ge)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    patience = 4
    wait = 0
    best = float('inf')
    for epoch in range(epochs):
        for step, ((x_wav, y_wav), (x_ge, y_ge)) in enumerate(zip(train_ds_wav2vec, train_ds_gemaps)):
            assert tf.reduce_all(y_wav == y_ge)
            with tf.GradientTape() as tape:
                y_pred = model(wav2vec_normalizer(x_wav), gemaps_normalizer(x_ge), training=True)
                loss_value = tf.keras.losses.sparse_categorical_crossentropy(y_wav, y_pred, from_logits=True)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            print(f"\rEpoch {epoch + 1}/{epochs}, Step {step}, Loss {np.mean(loss_value)}", end='', flush=True)
        print()
        val_loss = test(model, val_ds_wav2vec, val_ds_gemaps)
        wait += 1
        if val_loss < best:
            best = val_loss
            wait = 0
            dir_to_save = '../../results/saved_models/paper/ravdess_paper'
            if not os.path.exists(dir_to_save):
                Path(dir_to_save).mkdir(parents=True, exist_ok=True)
            model.save_weights(dir_to_save)
        if wait >= patience:
            break

    return model


def test(model: tf.keras.Model, test_ds_wav2vec: tf.data.Dataset, test_ds_gemaps: tf.data.Dataset):
    global wav2vec_normalizer, gemaps_normalizer
    y_cum, y_pred_cum = [], []
    loss_fn = tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True)
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    for (x_wav, y_wav), (x_ge, y_ge) in zip(test_ds_wav2vec, test_ds_gemaps):
        assert tf.reduce_all(y_wav == y_ge)
        y_pred = model(wav2vec_normalizer(x_wav), gemaps_normalizer(x_ge), training=False)
        loss_fn.update_state(y_wav, y_pred)
        accuracy.update_state(y_wav, y_pred)
        y_cum += list(y_wav.numpy())
        y_pred_cum = np.append(y_pred_cum, tf.argmax(y_pred, 1).numpy())
    recall = recall_score(y_cum, y_pred_cum, average="macro")
    print(f"\rDev Loss {loss_fn.result()}, Accuracy: {accuracy.result()} Average recall: {recall}", end='', flush=True)
    print()
    return loss_fn.result()
