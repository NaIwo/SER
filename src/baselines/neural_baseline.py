import os
from pathlib import Path

import wav2vec_neural.models as models
from src.baselines.config_reader import config
from src.datasets import get_dataset_by_name


def training(dataset):
    use_wav2vec = True if dataset.data_status == 'raw_data' else False
    model = getattr(models, 'Wav2vecNormalize')(num_of_classes=dataset.number_of_classes)

    train_iterator = dataset.train_iterator(batch_size=config['data']['dataset']['batch-size'],
                                            prefetch=config['data']['dataset']['prefetch'])
    val_iterator = dataset.val_iterator(batch_size=config['data']['dataset']['batch-size'],
                                        prefetch=config['data']['dataset']['prefetch'])

    # training model
    model = models.train_wav2vec(model, train_iterator, val_iterator,
                                 epochs=config['model']['wav2vec2']['train-epochs'],
                                 use_wav2vec=use_wav2vec)

    dir_to_save = config['model']['wav2vec2']['save-dir']
    if not os.path.exists(dir_to_save):
        Path(dir_to_save).mkdir(parents=True, exist_ok=True)
    model.clf.save_weights(dir_to_save)


def testing(dataset):
    batch_size = config['data']['dataset']['batch-size']

    use_wav2vec = True if dataset.data_status == 'raw_data' else False

    model = getattr(models, 'Wav2vecNormalize')(num_of_classes=dataset.number_of_classes)
    model.clf.load_weights(config['model']['wav2vec2']['save-dir'])

    test_iterator = dataset.test_iterator(batch_size=batch_size, prefetch=config['data']['dataset']['prefetch'])

    print("\nTest results: ", end="")
    models.test_wav2vec(model, test_iterator, use_wav2vec=use_wav2vec)


if __name__ == '__main__':
    Dataset = get_dataset_by_name(config['data']['dataset']['name'])
    dataset = Dataset(total_length=config['data']['dataset']['desired-length'],
                      padding_value=config['data']['dataset']['padding-value'],
                      train_size=config['data']['dataset']['train-size'],
                      test_size=config['data']['dataset']['test-size'],
                      val_size=config['data']['dataset']['val-size'],
                      data_status=config['data']['source-name'],
                      seed=config['data']['dataset']['shuffle-seed'],
                      resample_training_set=config['data']['dataset']['resample-training-set'])

    if config['model']['wav2vec2']['mode'] == 'training':
        training(dataset)

    testing(dataset)
