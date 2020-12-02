from argparse import Namespace
from typing import Callable, Optional
from omegaconf import OmegaConf
import submitit


class Launcher:
    @classmethod
    def run(cls, args):
        raise NotImplementedError()

    @classmethod
    def launch(cls, args):
        if args.slurm:
            cls.run_on_slurm(args)
        else:
            cls.run_locally(args)

    @classmethod
    def run_locally(cls, args):
        cls.run(args)

    @classmethod
    def run_on_slurm(cls, args):
        slurm_config = OmegaConf.load(args.slurm)
        nb_gpus = slurm_config.get("gpus_per_node", 1)
        mem_by_gpu = slurm_config.get("mem_by_gpu", 60)
        log_folder = slurm_config["log_folder"]

        executor = submitit.AutoExecutor(folder=log_folder)
        executor.update_parameters(slurm_partition=slurm_config.get("partition", ""),
                                   slurm_comment=slurm_config.get("comment", ""),
                                   slurm_constraint=slurm_config.get("gpu_type", ""),
                                   timeout_min=slurm_config.get("time_in_min", 30),
                                   nodes=slurm_config.get("nodes", 1),
                                   cpus_per_task=slurm_config.get("cpus_per_task", 10),
                                   tasks_per_node=nb_gpus,
                                   gpus_per_node=nb_gpus,
                                   mem_gb=mem_by_gpu * nb_gpus,)

        job = executor.submit(cls.run, args)
        print(f"{job.job_id}")