from paper_network.models.trainers import train, test
from paper_network.models.paper_clf import PaperNetwork
from src.baselines.config_reader import config
from src.datasets import get_dataset_by_name
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == '__main__':
    Dataset = get_dataset_by_name(config['data']['dataset']['name'])
    dataset_wav2vec = Dataset(data_status='960base_wav2vec_hidden_states',
                              seed=42,
                              padding_value=0.0,
                              total_length=250,
                              resample_training_set=config['data']['dataset']['resample-training-set'])

    dataset_gemaps = Dataset(data_status='ravdess_gemaps_llds_002_001',
                             seed=42,
                             padding_value=0.0,
                             total_length=500,
                             resample_training_set=config['data']['dataset']['resample-training-set'])

    model = PaperNetwork(dataset_wav2vec.number_of_classes)

    train_iterator_gemaps = dataset_gemaps.train_iterator(batch_size=config['data']['dataset']['batch-size'],
                                                          prefetch=config['data']['dataset']['prefetch'])
    val_iterator_gemaps = dataset_gemaps.val_iterator(batch_size=config['data']['dataset']['batch-size'],
                                                      prefetch=config['data']['dataset']['prefetch'])
    test_iterator_gemaps = dataset_gemaps.test_iterator(batch_size=config['data']['dataset']['batch-size'],
                                                        prefetch=config['data']['dataset']['prefetch'])

    train_iterator_wav2vec = dataset_wav2vec.train_iterator(batch_size=config['data']['dataset']['batch-size'],
                                                            prefetch=config['data']['dataset']['prefetch'])
    val_iterator_wav2vec = dataset_wav2vec.val_iterator(batch_size=config['data']['dataset']['batch-size'],
                                                        prefetch=config['data']['dataset']['prefetch'])
    test_iterator_wav2vec = dataset_wav2vec.test_iterator(batch_size=config['data']['dataset']['batch-size'],
                                                          prefetch=config['data']['dataset']['prefetch'])

    train(model, train_iterator_wav2vec, val_iterator_wav2vec, train_iterator_gemaps, val_iterator_gemaps, 100)
    test(model, test_iterator_wav2vec, test_iterator_gemaps)
