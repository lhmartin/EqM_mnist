import torch
from abc import ABC, abstractmethod
from train.utils import model_size_b, MiB
from tqdm import tqdm
import torch.nn as nn
from data.sampleable import GaussianConditionalProbabilityPath, IsotropicGaussian, GaussianEqM
from data.simulator import ConditionalVectorField
from model.mnist_unet import MNISTUNet

class Trainer(ABC):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @abstractmethod
    def get_train_loss(self, **kwargs) -> torch.Tensor:
        pass

    def get_optimizer(self, lr: float):
        return torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, num_epochs: int, device: torch.device, lr: float = 1e-3, **kwargs) -> torch.Tensor:
        # Report model size
        size_b = model_size_b(self.model)
        print(f'Training model with size: {size_b / MiB:.3f} MiB')
        
        # Start
        self.model.to(device)
        opt = self.get_optimizer(lr)
        self.model.train()
        losses = []
        # Train loop
        pbar = tqdm(enumerate(range(num_epochs)))
        for idx, epoch in pbar:
            opt.zero_grad()
            loss = self.get_train_loss(**kwargs)
            losses.append(loss.detach().cpu().item())
            loss.backward()
            opt.step()
            pbar.set_description(f'Epoch {idx}, loss: {loss.item():.3f}')

        # Finish
        self.model.eval()
        return losses


class CFGTrainer(Trainer):
    def __init__(self, path: GaussianConditionalProbabilityPath, model: MNISTUNet, eta: float, **kwargs):
        assert eta > 0 and eta < 1
        super().__init__(model, **kwargs)
        self.eta = eta
        self.path = path

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        # Step 1: Sample z,y from p_data
        z,y = self.path.p_data.sample(batch_size)
        
        # Step 2: Set each label to 10 (i.e., null) with probability eta
        mask = torch.rand(batch_size) < self.eta
        mask.to(z)
        y[mask] = 10.0
        
        # Step 3: Sample t and x
        t = torch.rand(batch_size, 1, 1, 1).to(z)
        x = self.path.sample_conditional_path(z, t)

        # Step 4: Regress and output loss
        ut_pred = self.model(x,t,y)
        ut_actual = self.path.conditional_vector_field(x,z,t)

        return torch.nn.functional.mse_loss(ut_pred, ut_actual)
    
    
class EqMTrainer(Trainer):
    def __init__(self, data_sampler: GaussianEqM, model: ConditionalVectorField, device: torch.device, **kwargs):
        super().__init__(model, **kwargs)
        self.data_sampler = data_sampler
        self.model = model
        self.device = device
        self.eta = 0.2

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        # Step 1: Sample gamma
        gamma = torch.rand(batch_size, 1, 1, 1).to(self.device)
        xg, target, y = self.data_sampler.sample_conditional_path(gamma)
        
        # Step 2: Set each label to 10 (i.e., null) with probability eta
        mask = torch.rand(batch_size) < self.eta
        mask.to(xg)
        y[mask] = 10.0
        
        # Step 3: Predict
        pred = self.model(xg, torch.zeros_like(gamma), y)
        
        # Step 4: Output loss
        loss = torch.nn.functional.mse_loss(pred, target)
        return loss