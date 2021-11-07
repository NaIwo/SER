import tensorflow as tf
import librosa
import numpy as np

from datasets.base_reader import DatasetReaderBase


class RavdessReader(DatasetReaderBase):
    def __init__(self, desired_sampling_rate=16000, train_size=0.7, val_size=0.2, total_length=None,
                 padding_value=None, data_status='raw_data'):
        super().__init__('Ravdess', train_size, val_size, total_length, padding_value, data_status)
        self.desired_sampling_rate = desired_sampling_rate
        self._construct_dataset()

    def _construct_dataset(self):
        paths = self._load_all_data_paths()
        file_paths_dataset = tf.data.Dataset.from_tensor_slices(paths)

        labels = self._get_labels_from_file_names(paths)

        self.number_of_ds_examples = len(labels)
        self.number_of_classes = len(set(labels))

        labels_dataset = tf.data.Dataset.from_tensor_slices(labels)

        self.full_dataset = tf.data.Dataset.zip((file_paths_dataset, labels_dataset))
        self._construct_train_test_split()

    def _get_labels_from_file_names(self, paths):
        """
        https://zenodo.org/record/1188976

        Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
        Vocal channel (01 = speech, 02 = song).
        Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
        Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
        Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
        Repetition (01 = 1st repetition, 02 = 2nd repetition).
        Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).
        """
        labels = list()
        for path in paths:
            label = path.split('-')[-5]
            labels.append(self._convert_to_proper_label(label))  # speech

        return labels

    @staticmethod
    def _convert_to_proper_label(label):
        return int(label) - 1

    def _load_audio_raw(self, file_path, label):
        audio, sample_rate = librosa.load(bytes.decode(file_path.numpy()), sr=self.desired_sampling_rate)
        if self.padding_value is None or self.total_length is None:
            return tf.convert_to_tensor(audio, dtype=tf.float64), label
        else:
            audio_length = len(audio)
            if audio_length > self.total_length:
                audio = audio[:self.total_length]
            elif audio_length < self.total_length:
                audio = np.pad(audio, pad_width=(0, self.total_length - audio_length), mode='constant',
                               constant_values=self.padding_value)
            return tf.convert_to_tensor(audio, dtype=tf.float64), label
