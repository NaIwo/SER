import os
from pathlib import Path

from mfcc_cnn.models import *
from src.baselines.mfcc_cnn.models.mfcc_classifiers import MfccCNN
from src.config_reader import config
from src.datasets import get_dataset_by_name


def train_model(model, train_set, val_set, save_dir, epochs=30):
    model = train_mfcc_cnn(model, train_set, val_set, epochs=epochs)
    dir_to_save = save_dir
    if not os.path.exists(dir_to_save):
        Path(dir_to_save).mkdir(parents=True, exist_ok=True)
    model.clf.save_weights(dir_to_save)
    return model


def test_model(model, test_set, weights_dir, load_weights_from_file=True):
    if load_weights_from_file:
        model.clf.load_weights(weights_dir)
    test_mfcc_cnn(model, test_set)


def main():
    dataset_props = config['data']['dataset']
    Dataset = get_dataset_by_name(config['data']['dataset']['name'])
    dataset = Dataset(desired_sampling_rate=dataset_props['desired-sampling-rate'],
                      total_length=dataset_props['desired-length'],
                      padding_value=dataset_props['padding-value'],
                      train_size=dataset_props['train-size'],
                      test_size=dataset_props['test-size'],
                      val_size=dataset_props['val-size'],
                      data_status=config['data']['source-name'],
                      train_test_seed=dataset_props['shuffle-seed'],
                      resample_training_set=dataset_props['resample-training-set'])
    model_props = config['model']['gemaps-mfcc']['mfcc']
    batch_size = model_props['batch-size']
    train_iterator = dataset.train_iterator(batch_size=batch_size)
    val_iterator = dataset.val_iterator(batch_size=batch_size)
    test_iterator = dataset.test_iterator(batch_size=batch_size)
    model = MfccCNN(dataset.number_of_classes, model_props['number-windows'], model_props['number-coefficients'])
    weights_dir = model_props['save-dir']

    if model_props['mode'] == 'training':
        model = train_model(model, train_iterator, val_iterator, weights_dir, model_props['train-epochs'])
    print("Test results:")
    test_model(model, test_iterator, weights_dir)


if __name__ == '__main__':
    main()
