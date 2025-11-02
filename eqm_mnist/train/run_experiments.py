# script to run all experiments
import glob
import os
import yaml
import argparse
from experiment_runner import ExperimentRunner

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_folder", type=str)
    args = parser.parse_args()
    configs = glob.glob(os.path.join(args.config_folder, "*.yml"))
    # also search subfolders
    configs.extend(glob.glob(os.path.join(args.config_folder, "*", "*.yml")))
    print(configs)
    for config_path in configs:
        if config_path is None:
            continue
        
        print(f"Running experiment {config_path}")  
        config = yaml.safe_load(open(config_path, "r"))
        runner = ExperimentRunner(ExperimentRunner.Config(**config))
        runner.run()
        print(f"Experiment {config_path} completed")
    print("All experiments completed")