import math
import numpy as np
import torch
import torchvision
import transformers
import PIL
from dataclasses import dataclass, field
from tqdm import tqdm
from typing import Tuple, List
import os


def preproces_image(image: PIL.Image.Image, image_size: int) -> torch.Tensor:
    image_processor = torchvision.transforms.Compose([
        torchvision.transforms.Resize((image_size, image_size)),
        torchvision.transforms.ToTensor(),
    ])
    return image_processor(image).unsqueeze(0)


def get_model() -> torch.nn.Module:
    model = torchvision.models.vgg19(
        weights=torchvision.models.VGG19_Weights.DEFAULT,
    ).features
    for param in model.parameters():
        param.requires_grad_(False)
    model.to("cuda")

    return model


def content_loss(content: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sum((content - target) ** 2) / 2


def gram_matrix(tensor: torch.Tensor) -> torch.Tensor:
    N, d, h, w = tensor.size()
    tensor = tensor.view(N, d, h * w)
    gram = torch.bmm(tensor, tensor.transpose(1, 2))

    return gram


def style_loss(style: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    _, d, h, w = style.size()
    style_gram = gram_matrix(style)
    target_gram = gram_matrix(target)

    return torch.sum((style_gram - target_gram) ** 2) / (2 * (d * h * w))


def get_features(image: torch.Tensor, model: torch.nn.Module, layers: dict) -> dict:
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[name] = x
    return features




@dataclass
class Config:
    content_weight: float = 1
    style_weight: float = 1e4
    lr: float = 0.003
    iterations: int = 2000
    device: str = "cuda"
    content_layers: dict = field(default_factory=lambda: {
        '19': 1.0,
        '21': 1.0,
        '28': 1.0,
    })
    style_layers: dict = field(default_factory=lambda: {
        '0': 1.0,
        '5':  1.0,
        '10': 1.0,
    })
    image_size: int = 224
    log_interval: int = 250
    verbose: bool = True

    def __str__(self):
        content_layers_str = ', '.join([f"'{k}': {v}" for k, v in self.content_layers.items()])
        style_layers_str = ', '.join([f"'{k}': {v}" for k, v in self.style_layers.items()])
        return f"""
Config:
content_weight: {self.content_weight}
style_weight: {self.style_weight}
lr: {self.lr}
iterations: {self.iterations}
device: {self.device}
content_layers: {{{content_layers_str}}}
style_layers: {{{style_layers_str}}}
image_size: {self.image_size}
log_interval: {self.log_interval}
verbose: {self.verbose}
        """

def run_style_transfer(content_path: str, style_path: str, config: Config) -> Tuple[np.ndarray, List[float], List[float], List[float]]:
    content = preproces_image(PIL.Image.open(content_path), config.image_size).to(config.device)
    style = preproces_image(PIL.Image.open(style_path), config.image_size).to(config.device)

    model = get_model()
    content_features = get_features(content, model, config.content_layers)
    style_features = get_features(style, model, config.style_layers)

    target = content.clone().requires_grad_(True).to(config.device)

    optimizer = torch.optim.Adam([target], lr=config.lr)

    content_losses = []
    style_losses = []
    total_losses = []

    for i in tqdm(range(config.iterations), file=None if config.verbose else open(os.devnull, 'w')):
        target_features = get_features(target, model, {**config.content_layers, **config.style_layers})

        content_loss_value = 0
        style_loss_value = 0
        for name, target_feature in target_features.items():
            if name in config.content_layers:
                content_loss_value += config.content_layers[name] * content_loss(content_features[name], target_feature)
            if name in config.style_layers:
                style_loss_value += config.style_layers[name] * style_loss(style_features[name], target_feature)

        total_loss = config.content_weight * content_loss_value + config.style_weight * style_loss_value

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        content_losses.append(content_loss_value.item())
        style_losses.append(style_loss_value.item())
        total_losses.append(total_loss.item())

        if i % config.log_interval == 0 and config.verbose:
            print(f"Iteration: {i}, Loss: {total_loss.item()}")

    # clip values to [0, 1]
    target = torch.clamp(target, 0, 1)
    return target.squeeze(0).permute(1, 2, 0).cpu().detach().numpy(), content_losses, style_losses, total_losses
