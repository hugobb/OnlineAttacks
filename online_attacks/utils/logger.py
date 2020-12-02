from dataclasses import dataclass
from omegaconf import MISSING
import uuid
import json
import os
from enum import Enum


@dataclass
class LoggerParams:
    save_dir: str = MISSING


class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return {type(obj).__name__: obj.name}
        return json.JSONEncoder.default(self, obj)


class Logger:
    def __init__(self, params: LoggerParams = LoggerParams()):
        self.params = params
        self.uuid = uuid.uuid4()
        self.path = os.path.join(self.params.save_dir, str(self.uuid))
        os.makedirs(self.path, exist_ok=True)

    def save_record(self, record, name="record"):
        filename = os.path.join(self.path, name + ".json")
        with open(filename, "w+") as f:
            json.dump(record, f, cls=EnumEncoder)
        print("Saved record under %s"%filename)
        
