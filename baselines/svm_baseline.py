from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np

from datasets import RavdessReader

class_labels = ["Neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]


def print_metrics(y_true, y_pred):
    print("Class labels: ", class_labels)
    print("Confusion matrix:", confusion_matrix(y_true, y_pred), sep="\n")
    print("Classification accuracy:", accuracy_score(y_true, y_pred))


def main():
    dataset = RavdessReader(desired_sampling_rate=16000,
                            total_length=80000,
                            padding_value=0.0,
                            train_size=0.9,
                            val_size=0.0,
                            data_status='mfcc')
    svm = SVC(C=500)
    standard_scaler = StandardScaler()
    x_train, y_train = dataset.get_numpy_dataset(dataset.train_dataset)
    x_test, y_test = dataset.get_numpy_dataset(dataset.test_dataset)
    standard_scaler.fit(np.concatenate([x_train, x_test]))
    x_train = standard_scaler.transform(x_train)
    x_test = standard_scaler.transform(x_test)
    svm.fit(x_train, y_train)
    y_pred = svm.predict(x_test)
    print_metrics(y_test, y_pred)


if __name__ == '__main__':
    main()
