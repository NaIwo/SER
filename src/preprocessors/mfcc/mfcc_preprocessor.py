import librosa.feature
from python_speech_features import mfcc
from typing import Callable, Optional
import numpy as np
import tensorflow as tf

from src.preprocessors.config_reader import config
from src.datasets import get_dataset_by_name
from src.datasets import BaseDataset
from src.preprocessors.preprocessor import Preprocessor


class MfccPreprocessor(Preprocessor):
    def __init__(self, dataset: BaseDataset, target_dir: str, reduce_func: Optional[Callable] = np.mean,
                 number_of_coefficients: int = 13, window_length: float = 0.025, window_step: float = 0.01,
                 window_function: Callable = np.hamming, expand_dimension: bool = False):
        super().__init__(dataset, target_dir, reduce_func)
        self.coef_number = number_of_coefficients
        self.window_length = window_length
        self.window_step = window_step
        self.window_function = window_function
        self.expand_dimension = expand_dimension

    def preprocess_single_example(self, example: tf.Tensor) -> np.ndarray:
        mfccs = mfcc(example.numpy(), samplerate=self.dataset.desired_sampling_rate,
                     numcep=self.coef_number, winlen=self.window_length, winstep=self.window_step,
                     winfunc=self.window_function)
        # mfccs = librosa.feature.mfcc(example.numpy(), sr=self.dataset.desired_sampling_rate, n_mfcc=self.coef_number,
        #                              win_length=int(self.dataset.desired_sampling_rate * self.window_length),
        #                              hop_length=int(self.dataset.desired_sampling_rate * self.window_step),
        #                              n_fft=512, window=self.window_function).T
        if self.agg is not None:
            mfccs = self.agg(mfccs, axis=0)
        if self.expand_dimension:
            mfccs = np.expand_dims(mfccs, axis=-1)
        return mfccs

    def save_single_example(self, target_path: str, preprocessed_example: np.ndarray):
        np.save(target_path, preprocessed_example)


def main():
    Dataset = get_dataset_by_name(config['data']['dataset']['name'])
    dataset = Dataset(desired_sampling_rate=config['data']['dataset']['original-sampling-rate'],
                      data_status='raw_data',
                      use_augmented_data=config['data']['dataset']['use-augmented-data'])
    preprocessor = MfccPreprocessor(dataset, config['data']['out-name'], reduce_func=None, expand_dimension=True)
    preprocessor.preprocess_data()


if __name__ == '__main__':
    main()
