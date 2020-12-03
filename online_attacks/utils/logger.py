from dataclasses import dataclass
from omegaconf import MISSING
import uuid
import json
import os
from enum import Enum
from omegaconf import OmegaConf
from typing import Optional


@dataclass
class LoggerParams:
    save_dir: str = MISSING
    exp_id: Optional[str] = None


class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return {type(obj).__name__: obj.name}
        return json.JSONEncoder.default(self, obj)


class Logger:
    def __init__(self, params: LoggerParams = LoggerParams()):
        if params.exp_id is None:
            params.exp_id = str(uuid.uuid4())
        self.uuid = params.exp_id
        self.path = os.path.join(params.save_dir, self.uuid)
        self.records = []

    def save_hparams(self, hparams):
        os.makedirs(self.path, exist_ok=True)
        filename = os.path.join(self.path, "hparams.yaml")
        OmegaConf.save(config=hparams, f=filename)

    def save_record(self, record, name="record"):
        os.makedirs(self.path, exist_ok=True)
        filename = os.path.join(self.path, name + ".json")
        with open(filename, "w+") as f:
            json.dump(record, f, cls=EnumEncoder, indent=4)
        print("Saved record under %s" % filename)

    def load_hparams(self):
        filename = os.path.join(self.path, "hparams.yaml")
        return OmegaConf.load(filename)

    def load_record(self, name="record"):
        filename = os.path.join(self.path, name + ".json")
        with open(filename, "r") as f:
            record = json.load(f)
        return record

    @staticmethod
    def list_all_records(path):
        list_paths = os.listdir(path)
        list_records = []
        for exp_id in list_paths:
            params = LoggerParams(save_dir=path, exp_id=exp_id)
            logger = Logger(params)
            list_records.append(logger)
        return list_records