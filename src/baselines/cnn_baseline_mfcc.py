import os
from pathlib import Path
from comet_ml import Experiment

from src.baselines.mfcc_gemaps_trainers import *
from src.baselines.mfcc_cnn.models.mfcc_classifiers import MfccCNN
from src.baselines.config_reader import config
from src.cometutils.experiment_helpers import *
from src.datasets import get_dataset_by_name


def train_model(model, train_set, val_set, save_dir, epochs=30):
    model = train(model, train_set, val_set, epochs=epochs)
    dir_to_save = save_dir
    if not os.path.exists(dir_to_save):
        Path(dir_to_save).mkdir(parents=True, exist_ok=True)
    model.clf.save_weights(dir_to_save)
    return model


def test_model(model, test_set, weights_dir, load_weights_from_file=True):
    if load_weights_from_file:
        model.clf.load_weights(weights_dir)
    return test(model, test_set)


def main():
    dataset_props = config['data']['dataset']

    if config['model']['gemaps-mfcc']['mfcc']['record-experiments']:
        exp_name = f"{dataset_props['name']} - {config['data']['source-name']} - baseline CNN"
        exp_description = f"Train set size - {dataset_props['train-size'] * 100}%, Test set size - {dataset_props['test-size'] * 100}%, augmentation - {dataset_props['use-augmented-data']}"
        exp = create_experiment(exp_name, exp_description)
    else:
        exp = None

    Dataset = get_dataset_by_name(config['data']['dataset']['name'])
    dataset = Dataset(total_length=dataset_props['desired-length'],
                      padding_value=dataset_props['padding-value'],
                      train_size=dataset_props['train-size'],
                      test_size=dataset_props['test-size'],
                      val_size=dataset_props['val-size'],
                      data_status=config['data']['source-name'],
                      seed=dataset_props['shuffle-seed'],
                      resample_training_set=dataset_props['resample-training-set'],
                      use_augmented_data=dataset_props['use-augmented-data'])
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
    losses, conf_matrix, metrics = test_model(model, test_iterator, weights_dir, True)
    if exp:
        record_metrics(exp, metrics, conf_matrix, dataset_props["name"])
        exp.end()


if __name__ == '__main__':
    main()
