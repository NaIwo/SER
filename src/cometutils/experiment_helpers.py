from typing import Optional

from comet_ml import Experiment


LABELS = {
    "RAVDESS": ["Neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"],
    "MELD": ['anger', 'disgust', 'sadness', 'joy', 'neutral', 'surprise', 'fear']
}


def create_experiment(name: str, description: Optional[str] = None, **kwargs):
    exp = Experiment(
        **kwargs,
        project_name="SER",
    )
    exp.set_name(name)
    exp.log_text(description)
    return exp


def record_metrics(exp: Experiment, metric_dict: dict, confusion_matrix: list[list], dataset):
    exp.log_metrics(metric_dict)
    exp.log_confusion_matrix(matrix=confusion_matrix, labels=LABELS[dataset])
