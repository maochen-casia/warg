import torch

from .warg.build_warg import build_warg

build_model_methods = {
    'WARG': build_warg
}

def build_model(config):
    model_name = config.name
    if model_name not in build_model_methods:
        raise ValueError(f"Model {model_name} not supported.")
    
    model = build_model_methods[model_name](config)
    print(f'Model {model_name} built successfully.')
    return model
