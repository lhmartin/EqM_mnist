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
import matplotlib.pyplot as plt

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
        # inference settings
        step_sizes: list[float] = [0.01, 0.02, 0.05, 0.1]
        nag_factors: list[float] = [0.1, 0.2, 0.4, 0.8]
        g_thresholds: list[int] = [10, 20, 50, 100]
        max_steps: list[int] = [50, 100, 200, 500]
        guidance_scales: list[float] = [0.5, 1.0, 5.0, 10.0, 100.0]

        
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
        
        samples, intermediate_steps = self.run_sampling(save_dir=self.config.save_dir)
        
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
        
    def get_inference_settings(self):
        """
        Returns a list of inference settings.
        """
        for step_size in self.config.step_sizes:
            for nag_factor in self.config.nag_factors:
                for g_threshold in self.config.g_thresholds:
                    for max_steps in self.config.max_steps:
                        for guidance_scale in self.config.guidance_scales:
                            yield {
                                "step_size": step_size,
                                "nag_factor": nag_factor,
                                "g_threshold": g_threshold,
                                "max_steps": max_steps,
                                "guidance_scale": guidance_scale
                            }

    def run_sampling(self, save_dir: str):
        """
        Samples from the trained model.
        Gets a batch of 50 samples of each digit, and an unguided sample.
        
        Loops through the inference settings defined in the config.
        Saves the outputs as images, video, and numpy arrays.
        
        """
                # Sample from the model
        sampler = EqMSampler(self.model, device=self.device)
        initial_samples = self.data_sampler.p_simple.sample(50)[0].to(self.device)
        for target_y in range(10):
            for inference_settings in self.get_inference_settings():
                samples, intermediate_steps = sampler.sample(initial_samples=initial_samples, 
                                                             **inference_settings,
                                                             target_y=target_y,
                                                             verbose=True)
                # save the samples
                inference_settings_str = "_".join([f"{k}={v}" for k, v in inference_settings.items()])
                os.makedirs(save_dir, exist_ok=True)
                plt.imsave(os.path.join(save_dir, f"samples_{target_y}_{inference_settings_str}.png"), samples.detach().cpu().numpy())
                np.save(os.path.join(save_dir, f"samples_{target_y}_{inference_settings_str}.npy"), samples.detach().cpu().numpy())
                intermediate_steps = torch.stack(intermediate_steps)
                np.save(os.path.join(save_dir, f"intermediate_steps_{target_y}_{inference_settings_str}.npy"), intermediate_steps.detach().cpu().numpy())
                print(f"Samples saved to {os.path.join(save_dir, f'samples_{target_y}_{inference_settings_str}.npy')}")
                print(f"Intermediate steps saved to {os.path.join(save_dir, f'intermediate_steps_{target_y}_{inference_settings_str}.npy')}")


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