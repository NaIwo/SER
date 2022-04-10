from typing import Optional

import tensorflow as tf
import numpy as np
from sklearn.metrics import recall_score, accuracy_score, f1_score, confusion_matrix


def train(model: tf.keras.Model, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, epochs: int):

    @tf.function(experimental_relax_shapes=True)
    def train_step(x, y):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y, y_pred)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    val_losses = list()
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_ds):
            loss_value = train_step(x, y)
            print(f"\rEpoch {epoch + 1}/{epochs}, Step {step}, Loss {np.mean(loss_value)}", end='', flush=True)
        print()
        val_loss = test(model, val_ds)
        val_losses.append(np.mean(val_loss))

    return model


def test(model: tf.keras.Model, test_ds: tf.data.Dataset):
    test_loss = np.array([])
    y_cum, y_pred_cum = [], []
    for x, y in test_ds:
        y_pred = model(x, training=False)
        loss_value = tf.keras.losses.sparse_categorical_crossentropy(y, y_pred)
        test_loss = np.append(test_loss, loss_value.numpy())
        y_cum += list(y.numpy())
        y_pred_cum = np.append(y_pred_cum, tf.argmax(y_pred, 1).numpy())
    recall = recall_score(y_cum, y_pred_cum, average="macro")
    accuracy = accuracy_score(y_cum, y_pred_cum)
    f1 = f1_score(y_cum, y_pred_cum, average="macro")
    conf_matrix = confusion_matrix(y_cum, y_pred_cum)
    metrics = {
        "accuracy": accuracy,
        "recall": recall,
        "f1": f1
    }
    print(f"\rDev Loss {np.mean(test_loss)}, Accuracy: {accuracy} Average recall: {recall}", end='', flush=True)
    print()
    return np.mean(test_loss), conf_matrix.tolist(), metrics
