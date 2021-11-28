import numpy as np
import opensmile
import tensorflow as tf

from src.config_reader import config
from src.datasets import get_dataset_by_name
from src.datasets import BaseDataset
from src.preprocessors.preprocessor import Preprocessor


class GemapsPreprocessor(Preprocessor):
    def __init__(self, dataset: BaseDataset, target_dir: str, gemaps_type, gemaps_level):
        super().__init__(dataset, target_dir)
        self.gemaps_extractor = opensmile.Smile(feature_set=gemaps_type, feature_level=gemaps_level)

    def preprocess_single_example(self, example: tf.Tensor) -> np.ndarray:
        return self.gemaps_extractor.process_signal(example.numpy(), self.dataset.desired_sampling_rate).to_numpy()[0]

    def save_single_example(self, target_path: str, preprocessed_example: np.ndarray) -> None:
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
                      train_test_seed=config['data']['dataset']['shuffle-seed'],
                      resample_training_set=config['data']['dataset']['resample-training-set'])
    preprocessor = GemapsPreprocessor(dataset, config['data']['source-name'], opensmile.FeatureSet.eGeMAPSv02,
                                      opensmile.FeatureLevel.Functionals)
    preprocessor.preprocess_data()


if __name__ == '__main__':
    main()
