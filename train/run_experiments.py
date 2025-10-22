# script to run all experiments
import glob
import yaml
from experiment_runner import ExperimentRunner

if __name__ == "__main__":
    
    configs = glob.glob("configs/*/*.yml")
    for config in configs:
        config = yaml.safe_load(open(config, "r"))
        runner = ExperimentRunner(ExperimentRunner.Config(**config))
        runner.run()