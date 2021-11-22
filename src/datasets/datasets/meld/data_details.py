from typing import List, Type, TypeVar
from envyaml import EnvYAML
from pathlib import Path
import os

from src.config_reader import ConfigReader

T = TypeVar('T', bound='DataLabels')

datasets_path: Path = Path(os.path.dirname(os.path.realpath(__file__))).parent.parent
metadata: EnvYAML = ConfigReader.read_config(
    os.path.join(os.path.join(datasets_path, 'datasets_files', 'MELD'), 'meld_metadata.yml'))

emotion2id = {
    'anger': 0,
    'disgust': 1,
    'sadness': 2,
    'joy': 3,
    'neutral': 4,
    'surprise': 5,
    'fear': 6
}


class PathDetails:
    def __init__(self, path: str):
        self.file_name: str = Path(path).stem

        self.dialogue_id: str = metadata[self.file_name]['Dialogue_ID']
        self.emotion: str = metadata[self.file_name]['Emotion']
        self.end_time: str = metadata[self.file_name]['EndTime']
        self.episode: str = metadata[self.file_name]['Episode']
        self.season: str = metadata[self.file_name]['Season']
        self.sentiment: str = metadata[self.file_name]['Sentiment']
        self.speaker: str = metadata[self.file_name]['Speaker']
        self.sr_no: str = metadata[self.file_name]['SrNo']
        self.start_time: str = metadata[self.file_name]['StartTime']
        self.utterance: str = metadata[self.file_name]['Utterance']
        self.utterance_id: str = metadata[self.file_name]['Utterance_ID']

    @property
    def proper_label(self) -> int:
        return emotion2id[self.emotion]


class DataLabels:
    def __init__(self, path_details: List[PathDetails]):
        self.path_details: List[PathDetails] = path_details

    @classmethod
    def from_paths(cls: Type[T], paths: List) -> T:
        details = list()
        for path in paths:
            detail = PathDetails(path)
            details.append(detail)  # speech
        return cls(details)

    @property
    def labels(self):
        labels = list()
        for details in self.path_details:
            labels.append(details.proper_label)
        return labels
