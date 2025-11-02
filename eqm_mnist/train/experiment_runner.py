from typing import Literal, Callable
import os
import numpy as np
from pydantic import BaseModel
from eqm_mnist.data.sampleable import GaussianConditionalProbabilityPath, GaussianEqM, MNISTSampler
from eqm_mnist.model.mnist_unet import MNISTUNet
from eqm_mnist.sampler.eqm_sampler import EqMSampler
from eqm_mnist.train.c_functions import LinearC, PiecewiseC, TruncatedDecayC
from eqm_mnist.train.trainer import EqMTrainer

import torch
import yaml
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
        # model hyperparams
        channels: list[int] = [32, 64, 128]
        num_residual_layers: int = 2
        t_embed_dim: int = 40
        y_embed_dim: int = 40
        
    def __init__(self, config: 'Config'):
        self.config = config
        self.c_function = get_c_function(config)
        self.data_sampler = GaussianEqM(
            p_data=MNISTSampler(),
            p_simple_shape=P_SIMPLE_SHAPE,
            grad_magnitude_func=self.c_function,
            grad_multiplier=config.grad_multiplier
        )
        self.model = MNISTUNet(channels=config.channels, 
                               num_residual_layers=config.num_residual_layers, 
                               t_embed_dim=config.t_embed_dim, 
                               y_embed_dim=config.y_embed_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eta = config.eta
        self.trainer = EqMTrainer(data_sampler=self.data_sampler, model=self.model, device=self.device, eta=self.eta)
    
    def run(self):
        # Move model and data to device
        self.trainer.model.to(self.device)
        self.trainer.data_sampler.to(self.device)
        
        save_dir = os.path.join(self.config.save_dir, self.config.experiment_name)
        losses = self.trainer.train(num_epochs=self.config.num_epochs, 
                                    device=self.device, 
                                    lr=self.config.lr, 
                                    batch_size=self.config.batch_size)
        
        # Sample from the model
        sampler = EqMSampler(self.model, device=self.device)
        initial_samples = self.data_sampler.p_simple.sample(50)[0].to(self.device)
        samples, intermediate_steps = sampler.sample(initial_samples=initial_samples, 
                                 step_size=0.01, 
                                 nag_factor=0.4, 
                                 g_threshold=20,
                                 max_steps=200, 
                                 target_y=5, 
                                 guidance_scale=5.0,
                                 verbose=True)
        
        # save the samples
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "samples.npy"), samples.detach().cpu().numpy())
        intermediate_steps = torch.stack(intermediate_steps)
        np.save(os.path.join(save_dir, "intermediate_steps.npy"), intermediate_steps.detach().cpu().numpy())
        print(f"Samples saved to {os.path.join(save_dir, "samples.npy")}")
        print(f"Intermediate steps saved to {os.path.join(save_dir, "intermediate_steps.npy")}")
        
        # Save results
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "losses.npy"), losses)
        print(f"Results saved to {os.path.join(save_dir, "losses.npy")}")
        
        # save the model
        torch.save(self.model.state_dict(), os.path.join(save_dir, "model.pth"))
        print(f"Model saved to {os.path.join(save_dir, "model.pth")}")
        
        # save the config as a yaml file
        with open(os.path.join(save_dir, "config.yaml"), "w") as f:
            yaml.dump(self.config.model_dump(), f)
        print(f"Config saved to {os.path.join(save_dir, "config.yaml")}")


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