import os
from pathlib import Path

from comet_ml import Experiment

from src.baselines.mfcc_gemaps_trainers import *
from src.baselines.gemaps_cnn.models.gemaps_classifiers import *
from src.baselines.config_reader import config
from src.cometutils.experiment_helpers import *
from src.datasets import get_dataset_by_name


def train_model(model, train_set, val_set, save_dir, epochs=30, actor_normalization=False):
    if actor_normalization:
        model = train_actor_normalization(model, train_set, val_set, epochs=epochs)
    else:
        model = train(model, train_set, val_set, epochs=epochs)
    dir_to_save = save_dir
    if not os.path.exists(dir_to_save):
        Path(dir_to_save).mkdir(parents=True, exist_ok=True)
    model.save_weights(dir_to_save)
    return model


def test_model(model, test_set, weights_dir, load_weights_from_file=True, actor_normalization=False):
    if load_weights_from_file:
        model.load_weights(weights_dir)
    if actor_normalization:
        return test_actor_normalization(model, test_set)
    return test(model, test_set)


def main():
    dataset_props = config['data']['dataset']
    if config['model']['gemaps-mfcc']['gmaps']['record-experiments']:
        exp_name = f"TEST"
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
                      use_augmented_data=dataset_props['use-augmented-data'],
                      keep_actor_data=dataset_props['keep-actor-data'])
    model_props = config['model']['gemaps-mfcc']['gmaps']
    batch_size = model_props['batch-size']
    # for i in dataset.train_dataset:
    #     print(i[0], len(i))
    #     break
    if dataset.keep_actor_data:
        dataset.train_dataset = dataset.create_dataset_with_statistics(dataset.train_dataset)
        dataset.val_dataset = dataset.create_dataset_with_statistics(dataset.val_dataset)
        dataset.test_dataset = dataset.create_dataset_with_statistics(dataset.test_dataset)
    # for i in dataset.train_dataset:
    #     print(i[0], len(i))
    #     break
    train_iterator = dataset.train_iterator(batch_size=batch_size)
    val_iterator = dataset.val_iterator(batch_size=batch_size)
    test_iterator = dataset.test_iterator(batch_size=batch_size)
    weights_dir = model_props['save-dir']
    if not dataset.keep_actor_data:
        model = GemapsCNN(dataset.number_of_classes, 1, 4)
        if model_props['mode'] == 'training':
            model = train_model(model, train_iterator, val_iterator, weights_dir, model_props['train-epochs'], dataset.keep_actor_data)
        print("Test results:")
        losses, conf_matrix, metrics = test_model(model, test_iterator, weights_dir, True, dataset.keep_actor_data)
    else:
        model = GemapsCNNWithNormalization(dataset.number_of_classes)
        if model_props['mode'] == 'training':
            model = train_model(model, train_iterator, val_iterator, weights_dir, model_props['train-epochs'], dataset.keep_actor_data)
        print("Test results:")
        losses, conf_matrix, metrics = test_model(model, test_iterator, weights_dir, True, dataset.keep_actor_data)
    if exp:
        record_metrics(exp, metrics, conf_matrix, config['data']['dataset']['name'])
        exp.end()


if __name__ == '__main__':
    main()
