import numpy as np
import opensmile

from datasets import RavdessReader
from preprocessors.preprocessor import Preprocessor


class GemapsPreprocessor(Preprocessor):
    def __init__(self, dataset, target_dir, gemaps_type, gemaps_level):
        super().__init__(dataset, target_dir)
        self.gemaps_extractor = opensmile.Smile(feature_set=gemaps_type, feature_level=gemaps_level)

    def preprocess_single_example(self, example):
        return self.gemaps_extractor.process_signal(example, self.dataset.desired_sampling_rate).to_numpy()[0]

    def save_single_example(self, target_path, preprocessed_example):
        np.save(target_path, preprocessed_example)


def main():
    dataset = RavdessReader(desired_sampling_rate=16000,
                            total_length=80000,
                            padding_value=0.0,
                            train_size=0.0,
                            val_size=1.0)  # because val is not shuffled
    preprocessor = GemapsPreprocessor(dataset, "egemaps2_low_level", opensmile.FeatureSet.eGeMAPSv02, opensmile.FeatureLevel.LowLevelDescriptors)
    preprocessor.preprocess_data()


if __name__ == '__main__':
    main()
