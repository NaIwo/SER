from src.preprocessors.config_reader import config
from src.datasets import get_dataset_by_name
from src.preprocessors.preprocessor import Preprocessor
from src.preprocessors.wav2vec import Wav2VecModel
from src.datasets import BaseDataset

import tensorflow as tf
import numpy as np


class Wav2VecPreprocessor(Preprocessor):
    def __init__(self, dataset: BaseDataset, target_dir: str, wav2vec_model: Wav2VecModel):
        super().__init__(dataset, target_dir)
        self.wav2vec = wav2vec_model

    def preprocess_single_example(self, example: tf.Tensor) -> tf.Tensor:
        return self.wav2vec(tf.expand_dims(example, axis=0))

    def save_single_example(self, target_path: str, preprocessed_example: tf.Tensor):
        np.save(target_path, preprocessed_example.numpy())


if __name__ == '__main__':
    Dataset = get_dataset_by_name(config['data']['dataset']['name'])
    dataset = Dataset(desired_sampling_rate=config['data']['dataset']['original-sampling-rate'],
                      data_status='raw_data')
    preprocessor = Wav2VecPreprocessor(dataset, config['data']['out-name'], Wav2VecModel(dataset.number_of_classes))
    preprocessor.preprocess_data()
