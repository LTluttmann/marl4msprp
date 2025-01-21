import os
import torch
import pyrootutils
import numpy as np

root = pyrootutils.find_root(__file__, indicator=".gitignore")
data_path = os.path.join(root, "data_test/luttmann/")
instance_paths = [os.path.join(data_path, x) for x in os.listdir(data_path)]

solutions = {}

if __name__ == "__main__":
    for instance_path in instance_paths:
        instance_name = instance_path.split("/")[-1]
        solution_path = os.path.join(instance_path, "solutions.pth")

        solution = torch.load(solution_path)

        opt_sol = {k:v for k,v in solution.items() if v.is_optimal}

        frac_opt = len(opt_sol) / len(solution)
        avg_runtime = np.mean([x.runtime for x in solution.values()])
        avg_reward = np.mean([x.reward for x in solution.values()])

        solutions[instance_name] = {
            "frac_opt": frac_opt,
            "avg_runtime": avg_runtime,
            "avg_reward": avg_reward,
        }

    print(solutions)