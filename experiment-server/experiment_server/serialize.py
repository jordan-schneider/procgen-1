import dataclasses
import json
from typing import Any

import numpy as np


class Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif dataclasses.is_dataclass(obj):
            return self.encode(dataclasses.asdict(obj))
        return json.JSONEncoder.default(self, obj)


def serialize(obj: Any) -> str:
    return json.dumps(obj, cls=Encoder)
