import numpy as np
import opensmile
import tensorflow as tf

from datasets import RavdessReader, DatasetReaderBase
from preprocessors.preprocessor import Preprocessor


class GemapsPreprocessor(Preprocessor):
    def __init__(self, dataset: DatasetReaderBase, target_dir: str, gemaps_type, gemaps_level):
        super().__init__(dataset, target_dir)
        self.gemaps_extractor = opensmile.Smile(feature_set=gemaps_type, feature_level=gemaps_level)

    def preprocess_single_example(self, example: tf.Tensor) -> np.ndarray:
        return self.gemaps_extractor.process_signal(example.numpy(), self.dataset.desired_sampling_rate).to_numpy()[0]

    def save_single_example(self, target_path: str, preprocessed_example: np.ndarray) -> None:
        np.save(target_path, preprocessed_example)


def main():
    dataset = RavdessReader(desired_sampling_rate=16000,
                            total_length=80000,
                            padding_value=0.0,
                            train_size=0.0,
                            test_size=1.0)  # because test is not shuffled
    preprocessor = GemapsPreprocessor(dataset, "egemaps2", opensmile.FeatureSet.eGeMAPSv02, opensmile.FeatureLevel.Functionals)
    preprocessor.preprocess_data()


if __name__ == '__main__':
    main()
