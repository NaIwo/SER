import os

from wav2vec_neural.models import Wav2vecClassifierCNN, train_wav2vec, test_wav2vec
from datasets import RavdessReader


def training(dataset):
    use_wav2vec = True if dataset.data_status == 'raw_data' else False
    model = Wav2vecClassifierCNN(num_of_classes=dataset.number_of_classes)

    batch_size = 32

    train_iterator = dataset.train_iterator(batch_size=batch_size, prefetch=2)
    val_iterator = dataset.val_iterator(batch_size=batch_size, prefetch=2)

    # training model
    model = train_wav2vec(model, train_iterator, val_iterator, epochs=35, use_wav2vec=use_wav2vec)

    path = os.path.join('wav2vec_neural', 'wav2vec_clf_cnn_ravdess', 'wav2vec_clf_cnn')
    model.clf.save_weights(path)


def testing(dataset):
    path = os.path.join('wav2vec_neural', 'wav2vec_clf_cnn_ravdess', 'wav2vec_clf_cnn')
    batch_size = 16

    use_wav2vec = True if dataset.data_status == 'raw_data' else False

    model = Wav2vecClassifierCNN(num_of_classes=dataset.number_of_classes)
    model.clf.load_weights(path)

    test_iterator = dataset.test_iterator(batch_size=batch_size, prefetch=2)

    print("\nTest results: ", end="")
    test_wav2vec(model, test_iterator, use_wav2vec=use_wav2vec)


if __name__ == '__main__':
    dataset = RavdessReader(desired_sampling_rate=16000,
                            total_length=80000,
                            padding_value=0.0,
                            train_size=0.7,
                            test_size=0.1,
                            data_status='wav2vec_large_data_cnn')
    training(dataset)
    testing(dataset)
