from python_speech_features import mfcc
from typing import Callable, Optional
import numpy as np
import tensorflow as tf

from datasets import RavdessReader, DatasetReaderBase
from preprocessors.preprocessor import Preprocessor


class MfccPreprocessor(Preprocessor):
    def __init__(self, dataset: DatasetReaderBase, target_dir: str, reduce_func: Optional[Callable] = np.mean,
                 number_of_coefficients: int = 13, window_length: float = 0.025, window_step: float = 0.01,
                 window_function: Callable = np.hamming):
        super().__init__(dataset, target_dir, reduce_func)
        self.coef_number = number_of_coefficients
        self.window_length = window_length
        self.window_step = window_step
        self.window_function = window_function

    def preprocess_single_example(self, example: tf.Tensor) -> np.ndarray:
        return self.agg(mfcc(example.numpy(), samplerate=self.dataset.desired_sampling_rate,
                             numcep=self.coef_number, winlen=self.window_length, winstep=self.window_step,
                             winfunc=self.window_function), axis=0)

    def save_single_example(self, target_path: str, preprocessed_example: np.ndarray):
        np.save(target_path, preprocessed_example)


def main():
    dataset = RavdessReader(desired_sampling_rate=16000,
                            total_length=80000,
                            padding_value=0.0,
                            train_size=0.0,
                            test_size=1.0)  # because test is not shuffled
    preprocessor = MfccPreprocessor(dataset, "mfcc", number_of_coefficients=25)
    preprocessor.preprocess_data()


if __name__ == '__main__':
    main()
