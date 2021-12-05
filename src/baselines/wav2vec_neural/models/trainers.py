import tensorflow as tf
import numpy as np
from sklearn.metrics import recall_score, accuracy_score

from sklearn.metrics import recall_score


def train_wav2vec(model, train_ds, val_ds, epochs, use_wav2vec=True):
    optimizer = tf.keras.optimizers.Adam()
    val_losses = list()
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_ds):
            if use_wav2vec:
                data = model.wav2vec(x)
            else:
                data = x
            with tf.GradientTape() as tape:
                y_pred = model(data, training=True)
                y_pred = tf.squeeze(y_pred)
                loss_value = tf.keras.losses.sparse_categorical_crossentropy(y, y_pred)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            print(f"\rEpoch {epoch + 1}/{epochs}, Step {step}, Loss {np.mean(loss_value)}", end='', flush=True)
        print()
        val_loss = test_wav2vec(model, val_ds, use_wav2vec=use_wav2vec)
        val_losses.append(np.mean(val_loss))

    return model


def test_wav2vec(model, test_ds, use_wav2vec=True):
    test_loss = np.array([])
    y_cum, y_pred_cum = [], []
    for x, y in test_ds:
        if use_wav2vec:
            data = model.wav2vec(x)
        else:
            data = x
        y_pred = model(data, training=False)
        y_pred = tf.squeeze(y_pred)
        loss_value = tf.keras.losses.sparse_categorical_crossentropy(y, y_pred)
        test_loss = np.append(test_loss, loss_value.numpy())
        y_cum += list(y.numpy())
        y_pred_cum = np.append(y_pred_cum, tf.argmax(y_pred, 1).numpy())
    recall = recall_score(y_cum, y_pred_cum, average="macro")
    accuracy = accuracy_score(y_cum, y_pred_cum)
    print(f"\rDev Loss {np.mean(test_loss)}, Accuracy: {accuracy} Average recall: {recall}", end='',
          flush=True)
    print()
    return test_loss
