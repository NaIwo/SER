from python_speech_features import mfcc
import numpy as np

from datasets import RavdessReader
from preprocessors.preprocessor import Preprocessor


class MfccPreprocessor(Preprocessor):
    def __init__(self, dataset, target_dir):
        super().__init__(dataset, target_dir)

    def preprocess_single_example(self, example):
        return np.mean(mfcc(example.numpy(), samplerate=self.dataset.desired_sampling_rate, winfunc=np.hamming), axis=0)

    def save_single_example(self, target_path, preprocessed_example):
        np.save(target_path, preprocessed_example)


def main():
    dataset = RavdessReader(desired_sampling_rate=16000,
                            total_length=80000,
                            padding_value=0.0,
                            train_size=0.0,
                            val_size=1.0)  # because val is not shuffled
    preprocessor = MfccPreprocessor(dataset, "mfcc")
    preprocessor.preprocess_data()

if __name__ == '__main__':
    main()
