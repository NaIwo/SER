import warnings

import numpy as np
import opensmile
import tensorflow as tf

from src.config_reader import config
from src.datasets import get_dataset_by_name
from src.datasets import BaseDataset
from src.preprocessors.preprocessor import Preprocessor


class GemapsPreprocessor(Preprocessor):
    def __init__(self, dataset: BaseDataset, target_dir: str, gemaps_type, gemaps_level,
                 window_length: float = 0.06, window_step: float = 0.02):
        super().__init__(dataset, target_dir)
        if gemaps_level == opensmile.FeatureLevel.LowLevelDescriptors:
            self.window_size = window_length
            self.hop_size = window_step
        if self.window_size < 0.06:
            warnings.warn("Window lengths shorter than 0.06s except 0.02s not supported. Using 0.02s window with 0.01s step.")
        self.gemaps_extractor = opensmile.Smile(feature_set=gemaps_type, feature_level=gemaps_level, num_workers=None)

    def preprocess_single_example(self, example: tf.Tensor) -> np.ndarray:
        #windowed_signal = librosa.util.frame(example.numpy(), self.number_of_samples_per_window, self.hop_size, axis=0)
        if self.gemaps_extractor.feature_level == opensmile.FeatureLevel.LowLevelDescriptors:
            return self.__compute_llds(example)
        return self.__compute_functionals(example)

    def save_single_example(self, target_path: str, preprocessed_example: np.ndarray) -> None:
        np.save(target_path, preprocessed_example)

    def __compute_llds(self, example: tf.Tensor):
        numpy_signal = example.numpy()
        preprocessed_signal = []
        if self.window_size < 0.06:
            return self.gemaps_extractor(numpy_signal, self.dataset.desired_sampling_rate)[0].T
        for window_end in np.arange(self.window_size, stop=numpy_signal.size / self.dataset.desired_sampling_rate, step=self.hop_size):
            preprocessed_signal.append(
                self.gemaps_extractor.process_signal(numpy_signal, self.dataset.desired_sampling_rate,
                                                     start=window_end - self.window_size, end=window_end).to_numpy()[0])
        return np.array(preprocessed_signal)

    def __compute_functionals(self, example: tf.Tensor):
        return self.gemaps_extractor.process_signal(example.numpy(), self.dataset.desired_sampling_rate).to_numpy()[0]


def main():
    Dataset = get_dataset_by_name(config['data']['dataset']['name'])
    dataset = Dataset(desired_sampling_rate=config['data']['dataset']['desired-sampling-rate'],
                      total_length=config['data']['dataset']['desired-length'],
                      padding_value=config['data']['dataset']['padding-value'],
                      train_size=config['data']['dataset']['train-size'],
                      test_size=config['data']['dataset']['test-size'],
                      val_size=config['data']['dataset']['val-size'],
                      data_status='raw_data',
                      train_test_seed=config['data']['dataset']['shuffle-seed'],
                      resample_training_set=False,
                      use_augmented_data=True)
    preprocessor = GemapsPreprocessor(dataset, config['data']['source-name'], opensmile.FeatureSet.eGeMAPSv02,
                                      opensmile.FeatureLevel.LowLevelDescriptors, 0.02, 0.01)
    preprocessor.preprocess_data()


if __name__ == '__main__':
    main()
