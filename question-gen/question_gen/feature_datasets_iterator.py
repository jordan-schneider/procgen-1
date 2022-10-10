import pickle as pkl
from pathlib import Path
from typing import Iterable, List, Optional

from question_gen.trajectory_db import FeatureDataset


class FeatureDatasetsIterator(Iterable[FeatureDataset]):
    def __init__(self, paths: List[Path], max_length: Optional[int] = None) -> None:
        self.paths = paths
        self.max_length = max_length

    def __iter__(self):
        for path in self.paths:
            data: FeatureDataset = pkl.load(path.open("rb"))
            if self.max_length is not None:
                data = data.clip(max_length=self.max_length)
            yield data
