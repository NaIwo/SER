import tensorflow as tf
import numpy as np


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
    accuracy = np.array([])
    for x, y in test_ds:
        if use_wav2vec:
            data = model.wav2vec(x)
        else:
            data = x
        y_pred = model(data, training=False)
        loss_value = tf.keras.losses.sparse_categorical_crossentropy(y, y_pred)
        test_loss = np.append(test_loss, loss_value.numpy())
        acc = tf.keras.metrics.sparse_categorical_accuracy(y, y_pred)
        accuracy = np.append(accuracy, acc.numpy())
    print(f"\rDev Loss {np.mean(test_loss)}, Accuracy: {np.mean(accuracy)}", end='', flush=True)
    print()
    return test_loss
