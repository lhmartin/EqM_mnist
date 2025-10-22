from typing import Literal, Callable
import os
import numpy as np
from pydantic import BaseModel
from data.sampleable import GaussianEqM, MNISTSampler
from model.mnist_unet import MNISTUNet
from train.c_functions import LinearC, PiecewiseC, TruncatedDecayC
from train.trainer import CFGTrainer, EqMTrainer

import torch
import json
import os
import numpy as np

P_SIMPLE_SHAPE = [1, 32, 32]

class ExperimentRunner:
    
    class Config(BaseModel):
        experiment_name: str
        num_epochs: int = 1000
        lr: float = 1e-3
        batch_size: int = 250
        eta: float = 0.2
        c_function : Literal["linear", "truncated_decay", "piecewise"] = "linear"
        c_alpha: float = 0.8
        c_beta: float = 0.2
        grad_multiplier: float = 4.0
        save_dir: str = "results"
        
    def __init__(self, config: 'Config'):
        self.config = config
        self.c_function = get_c_function(config)
        self.data_sampler = GaussianEqM(p_data=MNISTSampler(), 
                                        p_simple_shape=P_SIMPLE_SHAPE, 
                                        grad_magnitude_func=self.c_function,
                                        grad_multiplier=config.grad_multiplier)
        self.model = MNISTUNet()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eta = config.eta
        self.trainer = EqMTrainer(data_sampler=self.data_sampler, model=self.model, device=self.device, eta=self.eta)
    
    def run(self):
        losses = self.trainer.train(num_epochs=self.config.num_epochs, device=self.device, lr=self.config.lr, batch_size=self.config.batch_size)
        # Save results
        os.makedirs(self.config.save_dir, exist_ok=True)
        np.save(os.path.join(self.config.save_dir, f"{self.config.experiment_name}.npy"), losses)
        print(f"Results saved to {os.path.join(self.config.save_dir, f'{self.config.experiment_name}.npy')}")
        
        # save the model
        torch.save(self.model.state_dict(), os.path.join(self.config.save_dir, f"{self.config.experiment_name}.pth"))
        print(f"Model saved to {os.path.join(self.config.save_dir, f'{self.config.experiment_name}.pth')}")
        
        # save the config
        with open(os.path.join(self.config.save_dir, f"{self.config.experiment_name}.json"), "w") as f:
            json.dump(self.config.model_dump(), f)
        print(f"Config saved to {os.path.join(self.config.save_dir, f'{self.config.experiment_name}.json')}")


def get_c_function(config: ExperimentRunner.Config) -> Callable:
    if config.c_function == "linear":
        return LinearC()
    elif config.c_function == "truncated_decay":
        return TruncatedDecayC(alpha=config.c_alpha)
    elif config.c_function == "piecewise":
        return PiecewiseC(alpha=config.c_alpha, beta=config.c_beta)
    else:
        raise ValueError(f"Invalid c_function: {config.c_function}")


if __name__ == "__main__":
    config = ExperimentRunner.Config()
    runner = ExperimentRunner(config)
    runner.run()