import yaml
import torch
import os

def load_config(config_path):
    """Loads configuration from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def save_model(model, path):
    """Saves a PyTorch model."""
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """Loads a PyTorch model."""
    model.load_state_dict(torch.load(path))
    return model
