import torch
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from typing import Tuple, List, Optional
from eqm_mnist.model.mnist_unet import MNISTUNet

class EqMSampler:
    """
    A class for sampling from the Equilibrium Model (EqM) using gradient-based optimization.
    """
    
    def __init__(self, model : MNISTUNet, device: Optional[str] = None):
        """
        Initialize the EqM sampler.
        
        Args:
            model: The neural network model to use for sampling
            device: Device to run the model on (e.g., 'cuda', 'cpu')
        """
        self.model = model.eval()
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.model.to(self.device)
        
    def cfe_grad(self, x: torch.Tensor, y: torch.Tensor, 
                 guidance_scale: float = 5.0) -> torch.Tensor:
        """
        Compute the classifier-free guidance gradient.
        
        Args:
            x: Input tensor
            y: Class labels
            guidance_scale: Scale factor for guidance
            
        Returns:
            Guided gradient tensor
        """
        t = torch.zeros(x.shape[0], 1, 1, 1, device=x.device)
        guided_grad = self.model(x, t, y)
        unguided_grad = self.model(x, t, torch.ones_like(y) * 10)
        return (1 - guidance_scale) * unguided_grad + guidance_scale * guided_grad
    
    def sample(self, initial_samples: torch.Tensor, step_size: float = 0.01, 
               nag_factor: float = 0.4, g_threshold: float = 20, 
               max_steps: int = 50, target_y: int = 1, 
               guidance_scale: float = 5.0, verbose: bool = True) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Sample from the EqM model using gradient-based optimization.
        
        Args:
            initial_samples: Initial samples to start optimization from
            step_size: Step size for gradient descent
            nag_factor: Nesterov accelerated gradient factor
            g_threshold: Gradient norm threshold for stopping
            max_steps: Maximum number of optimization steps
            target_y: Target class label
            guidance_scale: Scale factor for classifier-free guidance
            verbose: Whether to show progress bar
            
        Returns:
            Tuple of (final_samples, intermediate_steps)
        """
        self.model.eval()
        x = initial_samples.to(self.device)
        x_last = x.clone()
        y = torch.ones(x.shape[0]).to(self.device).int() * target_y
        t = torch.zeros(x.shape[0], 1, 1, 1).to(self.device)
        grad = self.cfe_grad(x, y, guidance_scale)
        
        step = 0
        norm_grad = torch.norm(grad)
        pbar = tqdm(range(max_steps)) if verbose else range(max_steps)
        sample_update_mask = torch.ones_like(y)
        intermediate_steps = []
        
        while sum(sample_update_mask) > 0 and step < max_steps:
            x_last = x.clone()
            x = x - step_size * grad * sample_update_mask.view(-1, 1, 1, 1)
            new_input = x + nag_factor * (x - x_last)
            grad = self.cfe_grad(new_input, y, guidance_scale)
            norm_grad = torch.linalg.matrix_norm(grad)
            sample_update_mask = norm_grad > g_threshold
            
            if verbose:
                pbar.set_description(f'Step {step} {sum(sample_update_mask).item()} / {x.shape[0]}, GN: {norm_grad.mean().item():.3f}')
                pbar.update(1)
            
            step += 1
            intermediate_steps.append(x_last.clone())
            
        if verbose:
            pbar.close()
            
        return x_last, intermediate_steps
    
    def plot_samples(self, samples: torch.Tensor, n_samples: int = 10, 
                     figsize: Tuple[int, int] = (10, 4), cmap: str = "gray") -> None:
        """
        Plot a grid of sampled images.
        
        Args:
            samples: Tensor of sampled images
            n_samples: Number of samples to plot
            figsize: Figure size
            cmap: Colormap for the images
        """
        n_samples = min(n_samples, samples.shape[0])
        rows = 2
        cols = 5
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten() if n_samples > 1 else [axes]
        
        for i in range(n_samples):
            img = samples[i].detach().cpu().view(1, 32, 32).squeeze().numpy()
            axes[i].imshow(img, cmap=cmap)
            axes[i].axis("off")
            
        # Hide unused subplots
        for i in range(n_samples, len(axes)):
            axes[i].axis("off")
            
        plt.tight_layout()
        plt.show()
    
    def sample_and_plot(self, initial_samples: torch.Tensor, n_plot: int = 10, 
                       **sampling_kwargs) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Sample from the model and plot the results.
        
        Args:
            initial_samples: Initial samples to start optimization from
            n_plot: Number of samples to plot
            **sampling_kwargs: Additional arguments for the sample method
            
        Returns:
            Tuple of (final_samples, intermediate_steps)
        """
        result, intermediate_steps = self.sample(initial_samples, **sampling_kwargs)
        self.plot_samples(result, n_plot)
        return result, intermediate_steps