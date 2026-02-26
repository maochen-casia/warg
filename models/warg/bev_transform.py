import torch
import torch.nn.functional as F


def get_inv_K(K: torch.Tensor) -> torch.Tensor:
    return torch.linalg.inv(K)

@torch.jit.script
def project_bev_graph(
                    node_coords: torch.Tensor,
                    node_depths: torch.Tensor,
                    K_left: torch.Tensor,
                    R_left2world: torch.Tensor,
                    t_left2world: torch.Tensor,
                    K_sat: torch.Tensor,
                    R_sat2world: torch.Tensor,
                    t_sat2world: torch.Tensor) -> torch.Tensor:
    """
    Input:
        node_coords: (B, N, 2)
        node_depths: (B, N)
        K/R/t: Standard camera parameters
    Output:
        (B, N, 2)
    Formulation:
        1. Back-project: P_src = depth * (uv_hom @ K_src^-T)
        2. Transform:    P_hom = P_src @ (R_src^T R_tgt K_tgt^T) + (t_src - t_tgt) @ (R_tgt K_tgt^T)
        3. Project:      pixels = P_hom_xy / P_hom_z
    """

    M_proj_sat = R_sat2world @ K_sat.transpose(-2, -1)  # (B, 3, 3)

    M_transform = R_left2world.transpose(-2, -1) @ M_proj_sat # (B, 3, 3)

    t_diff = t_left2world - t_sat2world # (B, 3)
    bias = t_diff.unsqueeze(1) @ M_proj_sat # (B, 1, 3)

    uv_hom = F.pad(node_coords, (0, 1), "constant", 1.0) # (B, N, 3)
    inv_K_left = torch.linalg.inv(K_left)
    
    ray_dir = uv_hom @ inv_K_left.transpose(-2, -1)
    
    vecs_transformed = ray_dir @ M_transform # (B, N, 3)
    p_hom = vecs_transformed * node_depths.unsqueeze(-1)
    
    p_hom.add_(bias)

    depth_sat = p_hom[..., 2:3] + 1e-8
    return p_hom[..., :2] / depth_sat


@torch.jit.script
def generate_candidate_graphs_by_translation(
                    node_coords: torch.Tensor,
                    node_depths: torch.Tensor,
                    K_left: torch.Tensor,
                    R_left2world: torch.Tensor,
                    t_left2world_init: torch.Tensor,
                    K_sat: torch.Tensor,
                    R_sat2world: torch.Tensor,
                    t_sat2world: torch.Tensor,
                    search_radius: torch.Tensor,
                    search_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Optimized via Broadcast Decomposition.
    Separates the projection into a Static term (original points) and a Dynamic term (translation offsets).
    
    Math:
        P_world_new = P_world_old + delta_t
        P_sat_new   = (P_world_old + delta_t - t_sat) @ R_sat
        P_hom_new   = P_sat_new @ K_sat^T
                    = (P_world_old - t_sat) @ R_sat @ K_sat^T  [Static Term]
                      + delta_t @ R_sat @ K_sat^T              [Dynamic Term]
    """
    B, N, _ = node_coords.shape
    device = node_coords.device
    
    M_proj_sat = R_sat2world @ K_sat.transpose(-2, -1) # (B, 3, 3)
    
    M_static = R_left2world.transpose(-2, -1) @ M_proj_sat # (B, 3, 3)

    uv_hom = F.pad(node_coords, (0, 1), "constant", 1.0)
    inv_K_left = torch.linalg.inv(K_left)
    ray_dir = uv_hom @ inv_K_left.transpose(-2, -1)
    
    vecs_static = ray_dir @ M_static
    term_static = vecs_static * node_depths.unsqueeze(-1) # (B, N, 3)
    
    t_diff = t_left2world_init - t_sat2world
    bias_static = t_diff.unsqueeze(1) @ M_proj_sat # (B, 1, 3)
    
    term_static.add_(bias_static) # (B, N, 3) representing P_hom of the initial graph

    # Generate Grid (S, 2)
    step_range = torch.linspace(-1.0, 1.0, steps=search_steps, device=device)
    # Note: 'ij' indexing matches the original logic if it used meshgrid then stack(x,y)
    grid_y, grid_x = torch.meshgrid(step_range, step_range, indexing='ij')
    offsets_norm = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2) # (S, 2)
    
    offsets_world_xy = offsets_norm.unsqueeze(0) * search_radius.view(B, 1, 1)
    offsets_world = F.pad(offsets_world_xy, (0, 1), "constant", 0.0) # (B, S, 3)
    
    # (B, S, 3) @ (B, 3, 3) -> (B, S, 3)
    term_dynamic = offsets_world @ M_proj_sat
    
    p_hom_total = term_static.unsqueeze(1) + term_dynamic.unsqueeze(2)
    
    depth_final = p_hom_total[..., 2:3] + 1e-8
    candidate_node_coords = p_hom_total[..., :2] / depth_final  # (B, S, N, 2)

    candidate_t_left2world = t_left2world_init.unsqueeze(1) + offsets_world # (B, S, 3)

    return candidate_t_left2world, candidate_node_coords
