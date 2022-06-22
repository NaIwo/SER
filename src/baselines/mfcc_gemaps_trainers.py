from typing import Optional, Tuple, List

import tensorflow as tf
import numpy as np
from sklearn.metrics import recall_score, accuracy_score, f1_score, confusion_matrix

from src.baselines.gemaps_cnn.models.gemaps_classifiers import GemapsCNNWithNormalization


def train(model: tf.keras.Model, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, epochs: int):

    @tf.function(experimental_relax_shapes=True)
    def train_step(x, y):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y, y_pred)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return tf.reduce_mean(loss)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    val_losses = list()
    for epoch in range(epochs):
        train_losses = list()
        for step, (x, y) in enumerate(train_ds):
            train_losses.append(train_step(x, y))
            print(f"\rEpoch {epoch + 1}/{epochs}, Step {step}, Loss {np.mean(train_losses)}", end='', flush=True)
        print()
        val_loss, _, _ = test(model, val_ds)
        val_losses.append(np.mean(val_loss))

    return model


def train_actor_normalization(model: GemapsCNNWithNormalization, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, epochs: int):

    @tf.function(experimental_relax_shapes=True)
    def train_step(x, y, stats):
        with tf.GradientTape(persistent=True) as tape:
            y_pred, mean_pred, std_pred = model((x, stats), training=True)
            clf_loss = tf.keras.losses.sparse_categorical_crossentropy(y, y_pred)
            stats_loss = tf.keras.losses.mean_squared_error(stats / 1000, tf.stack((mean_pred, std_pred), axis=1))
        grads = tape.gradient(clf_loss, model.cnn.trainable_weights)
        stats_grads = tape.gradient(stats_loss, model.mean_std_regressor.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.cnn.trainable_weights))
        optimizer.apply_gradients(zip(stats_grads, model.mean_std_regressor.trainable_weights))
        return tf.reduce_mean(clf_loss), tf.reduce_mean(stats_loss)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    val_clf_losses, val_regression_losses = list(), list()
    for epoch in range(epochs):
        clf_losses, regression_losses = list(), list()
        for step, (x, y, stats) in enumerate(train_ds):
            clf_loss, regression_loss = train_step(x, y, stats)
            clf_losses.append(clf_loss)
            regression_losses.append(regression_loss)
            print(f"\rEpoch {epoch + 1}/{epochs}, Step {step}, Clf loss {np.mean(clf_losses):.4f}, Regression loss {np.mean(regression_losses):.4f}", end='', flush=True)
        print()
        val_losses, _, _ = test_actor_normalization(model, val_ds)
        val_clf_losses.append(val_losses[0])
        val_regression_losses.append(val_losses[1])

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
    metrics, conf_matrix = compute_classification_metrics(y_cum, y_pred_cum)
    print(f"\rDev Loss {np.mean(test_loss)}, Accuracy: {metrics['accuracy']} Average recall: {metrics['recall']} Weighted F1: {metrics['f1']}", end='', flush=True)
    print()
    return np.mean(test_loss), conf_matrix.tolist(), metrics


def test_actor_normalization(model: tf.keras.Model, test_ds: tf.data.Dataset):
    clf_loss, stats_loss = np.array([]), np.array([])
    y_cum, y_pred_cum = [], []
    for x, y, stats in test_ds:
        y_pred, mean_pred, std_pred = model((x, stats), training=False)
        clf_loss_value = tf.keras.losses.sparse_categorical_crossentropy(y, y_pred)
        stats_loss_value = tf.keras.losses.mean_squared_error(stats / 1000, tf.stack((mean_pred, std_pred), axis=1))
        clf_loss, stats_loss = np.append(clf_loss, clf_loss_value.numpy()), np.append(stats_loss, stats_loss_value.numpy())
        y_cum += list(y.numpy())
        y_pred_cum = np.append(y_pred_cum, tf.argmax(y_pred, 1).numpy())
    metrics, conf_matrix = compute_classification_metrics(y_cum, y_pred_cum)
    print(f"\rDev Loss {np.mean(clf_loss)}, Accuracy: {metrics['accuracy']}, Average recall: {metrics['recall']}, Weighted F1: {metrics['f1']}, Statistic MSE: {np.mean(stats_loss)}", end='', flush=True)
    print()
    return (np.mean(clf_loss), np.mean(stats_loss)), conf_matrix.tolist(), metrics


def compute_classification_metrics(y: List, y_pred: List) -> Tuple[dict, np.ndarray]:
    recall = recall_score(y, y_pred, average="macro")
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="weighted")
    conf_matrix = confusion_matrix(y, y_pred)
    metrics = {
        "accuracy": accuracy,
        "recall": recall,
        "f1": f1
    }
    return metrics, conf_matrix
