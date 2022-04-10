from typing import List, Type, TypeVar
from pathlib import Path
from functools import cached_property

T = TypeVar('T', bound='DataLabels')

emotion2id = {
    'angry': 0,
    'excited': 1,
    'happy': 2,
    'neutral': 3,
    'sad': 4,
    'anxious': 5,
    'apologetic': 6,
    'assertive': 7,
    'concerned': 8,
    'encouraging': 9,
}


class PathDetails:
    def __init__(self, path: str):
        """
        File naming rule: (Gender)(speaker.ID)_(Emotion)_(Sentence.ID)(session.ID)

        Gender (female, male).
        Speaker.ID (1 or 2).
        Emotion (see mapping above).
        SentenceID (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
        SessionID (1 or 2).
        """
        file_name: str = Path(path).stem
        # if file_name == 'desktop':
        #     return  # just Windows things...
        splitted_name: List = file_name.split('_')

        self.speaker_id: str = splitted_name[0]
        self.gender: str = self.speaker_id[:-1]
        self.number: int = int(self.speaker_id[-1])
        self.label: int = emotion2id[splitted_name[1]]
        self.statement: int = int(splitted_name[2][:-1])
        self.session_id: int = int(splitted_name[3])

    @cached_property
    def stratify_label(self) -> str:
        return str(self.label) + self.speaker_id


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

    @cached_property
    def labels(self):
        labels = list()
        for details in self.path_details:
            labels.append(details.label)
        return labels

    @cached_property
    def stratify_labels(self):
        stratify_labels = list()
        for details in self.path_details:
            stratify_labels.append(details.stratify_label)
        return stratify_labels
