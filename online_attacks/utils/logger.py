import uuid
import json
import os
from omegaconf import OmegaConf
from typing import Optional
import glob


def config_exists(config):
    if os.path.exists(config.save_dir):
        list_paths = os.listdir(config.save_dir)
    else:
        return None

    for exp_id in list_paths:
        logger = Logger(save_dir=config.save_dir, exp_id=exp_id)
        other_config = logger.load_hparams()
        other_config = OmegaConf.merge(config, other_config)
        if config == other_config:
            print("Found existing config with id=%s"%logger.exp_id)
            return logger.exp_id
    return None


class Logger:
    def __init__(self, save_dir: str, exp_id: Optional[str] = None):
        self.exp_id = exp_id
        if self.exp_id is None:
            self.exp_id = str(uuid.uuid4())
        self.path = os.path.join(save_dir, self.exp_id)
        os.makedirs(os.path.join(self.path, "runs"), exist_ok=True)

    def save_hparams(self, hparams):
        os.makedirs(self.path, exist_ok=True)
        filename = os.path.join(self.path, "hparams.yaml")
        OmegaConf.save(config=hparams, f=filename)

    def save_record(self, record):
        filename = os.path.join(self.path, "runs/%s.json"%(str(uuid.uuid4())))
        with open(filename, "w+") as f:
            json.dump(record, f, indent=4)

    def save_eval_results(self, eval_results, model_name, record_name):
        path = os.path.join(self.path, "eval", model_name)
        os.makedirs(path, exist_ok=True)
        filename = os.path.join(path, "%s.json"%record_name)
        with open(filename, "w+") as f:
            json.dump(eval_results, f, indent=4)

    def load_hparams(self):
        filename = os.path.join(self.path, "hparams.yaml")
        return OmegaConf.load(filename)

    def load_record(self, name):
        filename = os.path.join(self.path, "runs", name + ".json")
        with open(filename, "r") as f:
            record = json.load(f)
        return record

    def __len__(self):
        return len(self.list_all_records())

    def list_all_records(self):
        list_records = os.listdir(os.path.join(self.path, "runs"))
        return [os.path.splitext(filename)[0] for filename in list_records]

    @staticmethod
    def list_all_logger(path):
        list_exp_id = os.listdir(path)
        return list_exp_id

    @staticmethod
    def list_all_runs(path):
        return glob.glob(os.path.join(path, "*"))

    @staticmethod
    def load_eval_results(path):
        with open(path, "r") as f:
            record = json.load(f)
        return record

    def check_eval_results_exist(self, model_name, record_name):
        filename = os.path.join(self.path, "eval", model_name, "%s.json"%record_name)
        if os.path.exists(filename):
            return True
        return False

    def check_eval_done(self, model_name):
        path = os.path.join(self.path, "eval", model_name)
        if os.path.exists(path):
            list_dir = os.listdir(path)
        else:
            return False
        
        if len(list_dir) == len(self.list_all_records()):
            return True
        return False
