from typing import List, Type, TypeVar
from pathlib import Path
from functools import cached_property

T = TypeVar('T', bound='DataLabels')


class PathDetails:
    def __init__(self, path: str):
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
        file_name: str = Path(path).stem
        splitted_name: List = file_name.split('-')

        self.modality: str = splitted_name[0]
        self.vocal_channel: str = splitted_name[1]
        self.emotion: str = splitted_name[2]
        self.emotional_intensity: str = splitted_name[3]
        self.statement: str = splitted_name[4]
        self.replication: str = splitted_name[5]
        self.actor: int = int(splitted_name[6])

    @cached_property
    def stratify_label(self) -> str:
        return self.emotion + str(int(self.actor) % 2)

    @cached_property
    def proper_label(self) -> int:
        # if self.emotion != '01':
        #     return int(self.emotion) - 2
        return int(self.emotion) - 1


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
            labels.append(details.proper_label)
        return labels

    @cached_property
    def actors(self):
        actors = list()
        for details in self.path_details:
            actors.append(details.actor)
        return actors

    @cached_property
    def stratify_labels(self):
        stratify_labels = list()
        for details in self.path_details:
            stratify_labels.append(details.stratify_label)
        return stratify_labels
