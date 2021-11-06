from datasets import RavdessReader
from wav2vec import Wav2VecModel

import tensorflow as tf
import numpy as np
import os

if __name__ == '__main__':
    dataset = RavdessReader(desired_sampling_rate=16000,
                            total_length=80000,
                            padding_value=0.0,
                            train_size=0.0,
                            val_size=1.0)  # because val is not shuffled

    path_iterator = dataset.val_dataset
    wav2vec = Wav2VecModel(dataset.number_of_classes)

    for path, y in path_iterator:
        data, _ = dataset._load_audio_raw(path, y)
        raw_path_string = bytes.decode(path.numpy())
        raw_path_string = raw_path_string.replace('raw_data', 'wav2vec_data')

        path_to_save = raw_path_string.replace('.wav', '')
        dir_to_save = os.path.join(*path_to_save.split(os.sep)[:-1])

        wav2vec_out = wav2vec(tf.expand_dims(data, axis=0))

        if not (os.path.exists(dir_to_save)):
            os.mkdir(dir_to_save)
        np.save(path_to_save, wav2vec_out.numpy())
