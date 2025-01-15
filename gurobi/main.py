import os
import torch
import hydra
import logging
import pyrootutils
import numpy as np
from omegaconf import DictConfig

from marlprp.utils.dataset import get_file_dataloader
from marlprp.utils.config import TestParams

from gurobi.mip import solve

log = logging.getLogger(__name__)
pyrootutils.setup_root(__file__, indicator=".gitignore", pythonpath=True)

@hydra.main(version_base=None, config_path="../configs/", config_name="main")
def main(cfg: DictConfig):

    test_params = TestParams(**cfg.test)
    test_directory = os.path.dirname(test_params.data_dir["luttmann"])
    solution_file_name = os.path.join(test_directory, "solutions.pth")

    test_file_dls = get_file_dataloader(
        env=None,
        batch_size=1, 
        file_dir=test_params.data_dir
    )["luttmann"]
                
    solutions = {}
    objectives = []
    cnt = 0
    for batch in test_file_dls:
        for instance in batch:
            log.info(f"Start generating a solution for instance number {cnt}")
            solution = solve(instance, mipfocus=True, timeout=test_params.gurobi_timeout)
            solutions[cnt] = solution
            objectives.append(solution.reward)
            cnt += 1
    log.info(f"The average objective is {np.mean(objectives)}")
    torch.save(solutions, solution_file_name)

if __name__ == "__main__":

    main()


