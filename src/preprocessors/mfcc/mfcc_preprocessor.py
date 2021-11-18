from python_speech_features import mfcc
from typing import Callable, Optional
import numpy as np
import tensorflow as tf

from src.config_reader import config
from src.datasets import get_dataset_by_name
from src.datasets import BaseDataset
from src.preprocessors.preprocessor import Preprocessor


class MfccPreprocessor(Preprocessor):
    def __init__(self, dataset: BaseDataset, target_dir: str, reduce_func: Optional[Callable] = np.mean,
                 number_of_coefficients: int = 13, window_length: float = 0.025, window_step: float = 0.01,
                 window_function: Callable = np.hamming):
        super().__init__(dataset, target_dir, reduce_func)
        self.coef_number = number_of_coefficients
        self.window_length = window_length
        self.window_step = window_step
        self.window_function = window_function

    def preprocess_single_example(self, example: tf.Tensor) -> np.ndarray:
        mfccs = mfcc(example.numpy(), samplerate=self.dataset.desired_sampling_rate,
                     numcep=self.coef_number, winlen=self.window_length, winstep=self.window_step,
                     winfunc=self.window_function)
        if self.agg is None: return mfccs
        return self.agg(mfccs, axis=0)

    def save_single_example(self, target_path: str, preprocessed_example: np.ndarray):
        np.save(target_path, preprocessed_example)


def main():
    Dataset = get_dataset_by_name(config['data']['dataset']['name'])
    dataset = Dataset(desired_sampling_rate=config['data']['dataset']['desired-sampling-rate'],
                      total_length=config['data']['dataset']['desired-length'],
                      padding_value=config['data']['dataset']['padding-value'],
                      train_size=config['data']['dataset']['train-size'],
                      test_size=config['data']['dataset']['test-size'],
                      val_size=config['data']['dataset']['val-size'],
                      data_status='raw_data',
                      train_test_seed=config['data']['dataset']['shuffle-seed'])
    preprocessor = MfccPreprocessor(dataset, config['data']['source-name'],
                                    number_of_coefficients=config['model']['gemaps-mfcc']['number-coefficients'])
    preprocessor.preprocess_data()


if __name__ == '__main__':
    main()
