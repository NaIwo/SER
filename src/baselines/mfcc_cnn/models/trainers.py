import tensorflow as tf
import numpy as np


def train_mfcc_cnn(model: tf.keras.Model, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, epochs: int):
    optimizer = tf.keras.optimizers.Adam()
    val_losses = list()
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_ds):
            with tf.GradientTape() as tape:
                y_pred = model(x, training=True)
                loss_value = tf.keras.losses.sparse_categorical_crossentropy(y, y_pred)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            print(f"\rEpoch {epoch + 1}/{epochs}, Step {step}, Loss {np.mean(loss_value)}", end='', flush=True)
        print()
        val_loss = test_mfcc_cnn(model, val_ds)
        val_losses.append(np.mean(val_loss))

    return model


def test_mfcc_cnn(model: tf.keras.Model, test_ds: tf.data.Dataset):
    test_loss = np.array([])
    accuracy = np.array([])
    for x, y in test_ds:
        y_pred = model(x, training=False)
        loss_value = tf.keras.losses.sparse_categorical_crossentropy(y, y_pred)
        test_loss = np.append(test_loss, loss_value.numpy())
        acc = tf.keras.metrics.sparse_categorical_accuracy(y, y_pred)
        accuracy = np.append(accuracy, acc.numpy())
    print(f"\rDev Loss {np.mean(test_loss)}, Accuracy: {np.mean(accuracy)}", end='', flush=True)
    print()
    return test_loss
