"""
Training objective of the Online Decision Transformer (Zheng et al., 2022) for the discrete Skat actions.

The policy maximises the log-likelihood of the actions in the replay buffer subject to a lower bound on its entropy:
    min_theta  NLL(theta) - lambda * H(theta)
    min_lambda lambda * (H(theta) - target_entropy)
The temperature lambda is learned (as log lambda), such that the entropy stays around target_entropy and the policy
keeps exploring during the online fine-tuning.
"""

import math
from typing import Dict

import torch


class ODTUpdater:
    def __init__(self, model, learning_rate=1e-4, weight_decay=5e-4, warmup_steps=1000, grad_clip=0.25,
                 init_temperature=0.1, temperature_lr=1e-4, target_entropy=0.4):
        self.model = model
        self.grad_clip = grad_clip
        self.target_entropy = target_entropy

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda steps: min((steps + 1) / max(warmup_steps, 1), 1.0))

        self.log_temperature = torch.tensor(math.log(init_temperature), device=model.device, requires_grad=True)
        self.temperature_optimizer = torch.optim.Adam([self.log_temperature], lr=temperature_lr, betas=(0.9, 0.999))

    @property
    def temperature(self) -> float:
        return self.log_temperature.exp().item()

    def step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.train()
        batch = {key: value.to(self.model.device) for key, value in batch.items()}

        output = self.model.original_forward(**batch, return_dict=True)
        nll, entropy = output.loss, output.entropy

        loss = nll - self.log_temperature.exp().detach() * entropy

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        self.scheduler.step()

        temperature_loss = self.log_temperature.exp() * (entropy - self.target_entropy).detach()
        self.temperature_optimizer.zero_grad()
        temperature_loss.backward()
        self.temperature_optimizer.step()

        return {
            "loss": loss.item(),
            "nll": nll.item(),
            "entropy": entropy.item(),
            "temperature": self.temperature,
            "grad_norm": float(grad_norm),
            "learning_rate": self.scheduler.get_last_lr()[0],
        }
