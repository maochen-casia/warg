import torch

from .warg import WARG

def build_warg(config):
    device = torch.device(config.device)
    model = WARG(dino_model_name=config.dino_model_name,
                 num_nodes_per_scale=config.num_nodes_per_scale,
                 hid_dim=config.hid_dim,
                 match_dim=config.match_dim,
                 train_search_steps=config.train_search_steps,
                 eval_search_steps=config.eval_search_steps,
                 device=device)
    return model