from typing import Any
from omegaconf import OmegaConf
import submitit
from pathlib import Path
import datetime
import shutil
import os
import sys


def create_dedicated_execution_folder():
    """
    Create a specific folder to run the job on SLURM to make sure that each run
    is independent from any potential change in the code:
    - copy the current version of the code in that folder
    - copy the current version of the config in that folder
    - change the execution folder to that new folder
    """
    current_path = Path(".").absolute()
    run_id = str(datetime.datetime.now()).replace(" ", "-")
    run_folder = current_path.joinpath(".runs").joinpath(run_id).absolute()
    for folder in ["configs", "online_attacks", "data"]:
        shutil.copytree(current_path.joinpath(folder), run_folder.joinpath(folder))
    os.chdir(run_folder)
    sys.path = [str(run_folder)] + sys.path[1:] 


class SlurmLauncher:
    def __init__(self, run, checkpointing=False):
        self.run = run
        self.checkpointing = checkpointing

    def __call__(self, *args, **kwargs):
        self.run(*args, **kwargs)

    def checkpoint(self, *args: Any, **kwargs: Any) -> submitit.helpers.DelayedSubmission:
        if self.checkpointing:
            return submitit.helpers.DelayedSubmission(self, *args, **kwargs)  # submits to requeuing
        else:
            pass


class Launcher:
    def __init__(self, run=None, slurm="", checkpointing=False):
        self.run = run
        self.slurm = slurm
        self.checkpointing = checkpointing

    def launch(self, *args, **kwargs):
        if self.slurm:
            create_dedicated_execution_folder()
            self.run_on_slurm(*args, **kwargs)
        else:
            self.run_locally(*args, **kwargs)

    def run_locally(self, *args, **kwargs):
        self.run(*args, **kwargs)

    def run_on_slurm(self, *args, **kwargs):
        slurm_config = OmegaConf.load(self.slurm)
        nb_gpus = slurm_config.get("gpus_per_node", 1)
        mem_by_gpu = slurm_config.get("mem_by_gpu", 60)
        log_folder = slurm_config["log_folder"]

        executor = submitit.AutoExecutor(folder=log_folder)
        executor.update_parameters(slurm_partition=slurm_config.get("partition", ""),
                                   slurm_comment=slurm_config.get("comment", ""),
                                   slurm_constraint=slurm_config.get("gpu_type", ""),
                                   slurm_time=slurm_config.get("time_in_min", 30),
                                   timeout_min=slurm_config.get("time_in_min", 30),
                                   nodes=slurm_config.get("nodes", 1),
                                   cpus_per_task=slurm_config.get("cpus_per_task", 10),
                                   tasks_per_node=nb_gpus,
                                   gpus_per_node=nb_gpus,
                                   mem_gb=mem_by_gpu * nb_gpus,)

        slurm_launcher = SlurmLauncher(self.run, self.checkpointing)
        job = executor.submit(slurm_launcher, *args, **kwargs)
        print(f"{job.job_id}")