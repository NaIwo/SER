from python_speech_features import mfcc
import numpy as np
import tensorflow as tf

from datasets import RavdessReader, DatasetReaderBase
from preprocessors.preprocessor import Preprocessor


class MfccPreprocessor(Preprocessor):
    def __init__(self, dataset: DatasetReaderBase, target_dir: str):
        super().__init__(dataset, target_dir)

    def preprocess_single_example(self, example: tf.Tensor) -> np.ndarray:
        return np.mean(mfcc(example.numpy(), samplerate=self.dataset.desired_sampling_rate, winfunc=np.hamming), axis=0)

    def save_single_example(self, target_path: str, preprocessed_example: np.ndarray):
        np.save(target_path, preprocessed_example)


def main():
    dataset = RavdessReader(desired_sampling_rate=16000,
                            total_length=80000,
                            padding_value=0.0,
                            train_size=0.0,
                            test_size=1.0)  # because test is not shuffled
    preprocessor = MfccPreprocessor(dataset, "mfcc")
    preprocessor.preprocess_data()

if __name__ == '__main__':
    main()
